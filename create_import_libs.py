#!/usr/bin/env python3
"""
Create MSVC import libraries (.lib) from DLLs using dumpbin and lib.exe
"""
import subprocess
import os
import sys
import tempfile

# Paths
DUMPBIN_PATH = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\dumpbin.exe"
LIB_PATH = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\lib.exe"
DLLS = [
    r"C:\Program Files\AMD\ROCm\7.1\bin\libhipblas.dll",
    r"C:\Program Files\AMD\ROCm\7.1\bin\libhipblaslt.dll",
    r"C:\Program Files\AMD\ROCm\7.1\bin\amdhip64_7.dll",
]
OUTPUT_DIR = r"C:\msys64\opt\lib"

def dll_to_lib(dll_path, output_dir):
    """Convert DLL to .lib using dumpbin + lib.exe"""
    dll_name = os.path.basename(dll_path)
    base_name = os.path.splitext(dll_name)[0]
    lib_name = f"{base_name}.lib"
    lib_path = os.path.join(output_dir, lib_name)
    
    print(f"\nProcessing {dll_name} -> {lib_name}")
    
    # Create temporary .def file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.def', delete=False) as def_file:
        def_path = def_file.name
        def_file.write(f"LIBRARY {base_name}\nEXPORTS\n")
    
    try:
        # Extract exports from DLL using dumpbin
        print(f"  Extracting exports from DLL...")
        result = subprocess.run(
            [DUMPBIN_PATH, "/exports", dll_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"ERROR running dumpbin: {result.stderr}")
            return False
        
        # Parse dumpbin output to get exports
        exports = []
        for line in result.stdout.split('\n'):
            parts = line.split()
            if len(parts) >= 4 and parts[0].isdigit():
                # Format: "  1    0 XXXXXXXX  functionname"
                export_name = parts[3]
                if not export_name.startswith('_'):
                    exports.append(export_name)
        
        # Add exports to .def file
        with open(def_path, 'a') as def_file:
            for export in exports:
                def_file.write(f"  {export}\n")
        
        print(f"  Found {len(exports)} exports")
        
        # Create .lib using lib.exe
        print(f"  Creating import library...")
        result = subprocess.run(
            [LIB_PATH, f"/def:{def_path}", f"/out:{lib_path}"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"ERROR running lib.exe: {result.stderr}")
            return False
        
        if os.path.exists(lib_path):
            print(f"  Created {lib_path}")
            return True
        else:
            print(f"  ERROR: {lib_path} was not created")
            return False
            
    finally:
        if os.path.exists(def_path):
            os.unlink(def_path)

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    success = True
    for dll in DLLS:
        if os.path.exists(dll):
            if not dll_to_lib(dll, OUTPUT_DIR):
                success = False
        else:
            print(f"DLL not found: {dll}")
            success = False
    
    if success:
        print("\n✓ All import libraries created successfully")
        sys.exit(0)
    else:
        print("\n✗ Some import libraries failed to create")
        sys.exit(1)
