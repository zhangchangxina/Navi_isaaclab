#!/usr/bin/env python3
"""修复 SU17 USD 模型的质量和惯性矩参数

运行方式:
    ./isaaclab.sh -p scripts/fix_usd_inertia.py
"""

import argparse
from pxr import Usd, UsdPhysics, Gf

# SU17 真实参数 (基于官方规格估算)
SU17_MASS = 2.5  # kg
SU17_IXX = 0.03  # kg·m²
SU17_IYY = 0.03  # kg·m²
SU17_IZZ = 0.05  # kg·m²


def fix_usd_model(usd_path: str, output_path: str = None):
    """修复USD模型的物理参数"""
    
    # 打开USD文件
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"[ERROR] 无法打开 USD 文件: {usd_path}")
        return False
    
    print(f"[INFO] 打开 USD 文件: {usd_path}")
    
    # 遍历所有prim，找到刚体
    modified = False
    for prim in stage.Traverse():
        # 检查是否有 RigidBodyAPI
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            print(f"\n[INFO] 找到刚体: {prim.GetPath()}")
            
            # 获取或添加 MassAPI
            if not prim.HasAPI(UsdPhysics.MassAPI):
                UsdPhysics.MassAPI.Apply(prim)
                print(f"  [+] 添加 MassAPI")
            
            mass_api = UsdPhysics.MassAPI(prim)
            
            # 设置质量
            mass_attr = mass_api.GetMassAttr()
            if mass_attr:
                old_mass = mass_attr.Get()
                mass_attr.Set(SU17_MASS)
                print(f"  [*] 质量: {old_mass} -> {SU17_MASS} kg")
            else:
                mass_api.CreateMassAttr(SU17_MASS)
                print(f"  [+] 创建质量属性: {SU17_MASS} kg")
            
            # 设置惯性矩 (对角惯性张量)
            # USD 使用 diagonalInertia 属性
            inertia_attr = mass_api.GetDiagonalInertiaAttr()
            if inertia_attr:
                old_inertia = inertia_attr.Get()
                inertia_attr.Set(Gf.Vec3f(SU17_IXX, SU17_IYY, SU17_IZZ))
                print(f"  [*] 惯性矩: {old_inertia} -> ({SU17_IXX}, {SU17_IYY}, {SU17_IZZ}) kg·m²")
            else:
                mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(SU17_IXX, SU17_IYY, SU17_IZZ))
                print(f"  [+] 创建惯性矩属性: ({SU17_IXX}, {SU17_IYY}, {SU17_IZZ}) kg·m²")
            
            modified = True
    
    if not modified:
        print("[WARN] 未找到任何刚体，尝试查找根prim...")
        
        # 尝试找到名为 "body" 或根prim
        for prim in stage.Traverse():
            prim_name = prim.GetName().lower()
            if "body" in prim_name or prim.GetPath().pathString == "/SU17":
                print(f"\n[INFO] 尝试为 {prim.GetPath()} 添加物理属性")
                
                # 添加 RigidBodyAPI
                if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI.Apply(prim)
                    print(f"  [+] 添加 RigidBodyAPI")
                
                # 添加 MassAPI
                if not prim.HasAPI(UsdPhysics.MassAPI):
                    UsdPhysics.MassAPI.Apply(prim)
                
                mass_api = UsdPhysics.MassAPI(prim)
                mass_api.CreateMassAttr(SU17_MASS)
                mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(SU17_IXX, SU17_IYY, SU17_IZZ))
                print(f"  [+] 设置质量: {SU17_MASS} kg")
                print(f"  [+] 设置惯性矩: ({SU17_IXX}, {SU17_IYY}, {SU17_IZZ}) kg·m²")
                modified = True
                break
    
    if modified:
        # 保存修改
        if output_path:
            stage.Export(output_path)
            print(f"\n[SUCCESS] 已保存到: {output_path}")
        else:
            stage.Save()
            print(f"\n[SUCCESS] 已保存修改")
        return True
    else:
        print("[ERROR] 未能修改任何物理属性")
        return False


def print_usd_physics(usd_path: str):
    """打印USD模型的物理属性"""
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"[ERROR] 无法打开 USD 文件: {usd_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"USD 文件: {usd_path}")
    print(f"{'='*60}")
    
    for prim in stage.Traverse():
        has_physics = False
        
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            has_physics = True
            print(f"\n[刚体] {prim.GetPath()}")
        
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI(prim)
            mass = mass_api.GetMassAttr().Get()
            inertia = mass_api.GetDiagonalInertiaAttr().Get()
            print(f"  质量: {mass} kg")
            print(f"  惯性矩: {inertia} kg·m²")
        elif has_physics:
            print(f"  [WARN] 无 MassAPI，将使用默认值")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="修复 USD 模型的物理参数")
    parser.add_argument("--input", type=str, default="assets/Robots/SU17/SU17.usd",
                        help="输入 USD 文件路径")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 USD 文件路径 (默认覆盖原文件)")
    parser.add_argument("--print-only", action="store_true",
                        help="只打印物理属性，不修改")
    args = parser.parse_args()
    
    if args.print_only:
        print_usd_physics(args.input)
    else:
        # 先打印当前状态
        print("\n[当前状态]")
        print_usd_physics(args.input)
        
        # 修复
        print("\n[开始修复]")
        if fix_usd_model(args.input, args.output):
            print("\n[修复后状态]")
            print_usd_physics(args.output or args.input)
