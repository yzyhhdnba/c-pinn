#!/usr/bin/env python3
import os
import sys
import platform
import subprocess
import shutil

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("=" * 60)
    print("      C-PINN Configuration Wizard")
    print("=" * 60)
    print(f"Detected OS: {platform.system()} {platform.release()}")
    print("=" * 60)

def ask_mode():
    print("\nSelect Compilation Mode:")
    print("1) [Pure C++] Lightweight, CPU Only (Eigen3, No Torch)")
    print("   -> Best for: Logic testing, deployment on edge devices, basic PDE solving.")
    print("   -> Requires: CMake, Compiler, Eigen3, JSON.")
    print("\n2) [Torch Mode] High Performance (CPU/CUDA)")
    print("   -> Best for: Autograd, Complex NNs (ResNet/CNN), GPU Acceleration.")
    print("   -> Requires: LibTorch (C++ PyTorch).")
    
    while True:
        try:
            choice = input("\nEnter choice [1-2]: ")
            if choice == '1':
                return 'PURE'
            elif choice == '2':
                return 'TORCH'
        except KeyboardInterrupt:
            sys.exit(0)

def setup_torch_path():
    print("\n[Torch Mode Configuration]")
    default_path = os.environ.get("LIBTORCH_PATH", "")
    
    print("Please enter the absolute path to 'libtorch' directory.")
    if default_path:
        print(f"Found environment variable LIBTORCH_PATH={default_path}")
        use_default = input("Use this path? [Y/n]: ").lower()
        if use_default not in ['n', 'no']:
            return default_path
            
    while True:
        path = input("Path to libtorch: ").strip()
        # Basic validation
        if os.path.isdir(path) and (os.path.exists(os.path.join(path, "share", "cmake", "Torch")) or os.path.exists(os.path.join(path, "lib"))):
            return path
        print("Error: Invalid LibTorch path. Could not find 'share/cmake/Torch' or 'lib' inside.")
        print("Tip: Download from https://pytorch.org/get-started/locally/ (choose LibTorch C++)")

def build_project(mode, torch_path=None):
    build_dir = "build_pure" if mode == 'PURE' else "build_torch"
    
    if os.path.exists(build_dir):
        print(f"\nCleaning existing build directory: {build_dir}...")
        shutil.rmtree(build_dir)
    
    os.makedirs(build_dir)
    
    cmd = ["cmake", "-S", ".", "-B", build_dir]
    
    if mode == 'PURE':
        cmd.append("-DPINN_USE_TORCH=OFF")
        print("\nConfiguring for Pure C++ Mode...")
    else:
        cmd.append("-DPINN_USE_TORCH=ON")
        cmd.append(f"-DCMAKE_PREFIX_PATH={torch_path}")
        # Add CUDA check hint text
        cmd.append("-DTORCH_CUDA_ARCH_LIST='Common'") 
        print(f"\nConfiguring for Torch Mode (Path: {torch_path})...")

    cmd.append("-DCMAKE_BUILD_TYPE=Release")
    
    try:
        subprocess.check_call(cmd)
        
        print(f"\nBuilding project in {build_dir}...")
        # Get CPU count for parallel build
        cpu_count = os.cpu_count() or 4
        subprocess.check_call(["cmake", "--build", build_dir, "-j", str(cpu_count)])
        
        print("\n" + "="*60)
        print("BUILD SUCCESSFUL!")
        print("="*60)
        if mode == 'PURE':
            print("Run examples:")
            print(f"  ./{build_dir}/examples/example_pure_c_kdv")
        else:
            print("Run examples:")
            print(f"  ./{build_dir}/examples/example_burgers config/burgers_config.json")
            
    except subprocess.CalledProcessError:
        print("\nError: Build failed.")
        sys.exit(1)

def main():
    clear_screen()
    print_header()
    
    mode = ask_mode()
    
    torch_path = None
    if mode == 'TORCH':
        torch_path = setup_torch_path()
        
    build_project(mode, torch_path)

if __name__ == "__main__":
    main()
