#!/usr/bin/env python3
"""
Multi-platform build and test script for ViStreamASR

This script helps build and test ViStreamASR across different Python versions
and platforms locally before pushing to CI/CD.
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

class MultiPlatformBuilder:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.dist_dir = self.project_root / "dist"
        self.build_dir = self.project_root / "build"
        
    def clean_build_dirs(self):
        """Clean previous build artifacts."""
        print("🧹 Cleaning build directories...")
        for dir_path in [self.dist_dir, self.build_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                print(f"   Removed {dir_path}")
        
        # Clean egg-info
        for egg_info in self.project_root.glob("*.egg-info"):
            shutil.rmtree(egg_info)
            print(f"   Removed {egg_info}")
    
    def get_system_info(self):
        """Get current system information."""
        return {
            'platform': platform.platform(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'system': platform.system(),
            'architecture': platform.architecture()[0]
        }
    
    def run_command(self, cmd, cwd=None):
        """Run a command and return the result."""
        try:
            if cwd is None:
                cwd = self.project_root
            
            print(f"   Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, 
                cwd=cwd, 
                capture_output=True, 
                text=True, 
                check=True
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Command failed: {e}")
            print(f"   Error output: {e.stderr}")
            return False, e.stderr
    
    def check_dependencies(self):
        """Check if required build tools are available."""
        print("🔍 Checking build dependencies...")
        
        required_tools = ['python', 'pip']
        missing_tools = []
        
        for tool in required_tools:
            success, _ = self.run_command([tool, '--version'])
            if success:
                print(f"   ✅ {tool} available")
            else:
                missing_tools.append(tool)
                print(f"   ❌ {tool} not found")
        
        if missing_tools:
            print(f"❌ Missing required tools: {missing_tools}")
            return False
        
        # Check if build module is available
        success, _ = self.run_command([sys.executable, '-m', 'build', '--help'])
        if not success:
            print("📦 Installing build tools...")
            success, _ = self.run_command([sys.executable, '-m', 'pip', 'install', 'build', 'wheel'])
            if not success:
                print("❌ Failed to install build tools")
                return False
        
        return True
    
    def build_source_distribution(self):
        """Build source distribution."""
        print("📦 Building source distribution...")
        success, output = self.run_command([sys.executable, '-m', 'build', '--sdist'])
        if success:
            print("   ✅ Source distribution built successfully")
            return True
        else:
            print("   ❌ Source distribution build failed")
            return False
    
    def build_wheel(self):
        """Build wheel distribution."""
        print("🎯 Building wheel distribution...")
        success, output = self.run_command([sys.executable, '-m', 'build', '--wheel'])
        if success:
            print("   ✅ Wheel distribution built successfully")
            return True
        else:
            print("   ❌ Wheel distribution build failed")
            return False
    
    def test_installation(self):
        """Test installation from built wheel."""
        print("🧪 Testing installation...")
        
        # Find the built wheel
        wheels = list(self.dist_dir.glob("*.whl"))
        if not wheels:
            print("   ❌ No wheel found to test")
            return False
        
        wheel_path = wheels[0]
        print(f"   Testing wheel: {wheel_path.name}")
        
        # Create a temporary environment for testing
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # Install in temporary location
            success, _ = self.run_command([
                sys.executable, '-m', 'pip', 'install', 
                '--target', temp_dir,
                '--no-deps',  # Don't install dependencies for quick test
                str(wheel_path)
            ])
            
            if success:
                print("   ✅ Wheel installation successful")
                return True
            else:
                print("   ❌ Wheel installation failed")
                return False
    
    def run_basic_tests(self):
        """Run basic functionality tests."""
        print("🔬 Running basic tests...")
        
        # Test import functionality
        test_script = """
import sys
sys.path.insert(0, 'src')

try:
    from streaming import StreamingASR
    from core import ASREngine
    print('✅ Imports successful')
    
    # Test basic initialization
    asr = StreamingASR(debug=False)
    print(f'✅ StreamingASR initialized: {asr.chunk_size_ms}ms chunks')
    
    engine = ASREngine(debug_mode=False)
    print('✅ ASREngine initialized')
    
    print('🎉 All basic tests passed!')
    
except Exception as e:
    print(f'❌ Test failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
        
        # Write test script to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            temp_script = f.name
        
        try:
            success, output = self.run_command([sys.executable, temp_script])
            if success:
                print("   ✅ Basic tests passed")
                print(f"   Output: {output.strip()}")
                return True
            else:
                print("   ❌ Basic tests failed")
                return False
        finally:
            os.unlink(temp_script)
    
    def build_all(self):
        """Build all distributions and run tests."""
        print("🚀 Starting multi-platform build process...")
        print("=" * 60)
        
        # Show system info
        sys_info = self.get_system_info()
        print("💻 System Information:")
        for key, value in sys_info.items():
            print(f"   {key}: {value}")
        print()
        
        # Check dependencies
        if not self.check_dependencies():
            return False
        
        # Clean previous builds
        self.clean_build_dirs()
        
        # Run basic tests first
        if not self.run_basic_tests():
            print("❌ Basic tests failed, aborting build")
            return False
        
        # Build distributions
        success = True
        
        if not self.build_source_distribution():
            success = False
        
        if not self.build_wheel():
            success = False
        
        if success:
            # Test installation
            if not self.test_installation():
                success = False
        
        # Show results
        print("\n" + "=" * 60)
        if success:
            print("🎉 Multi-platform build completed successfully!")
            print("\n📦 Built distributions:")
            if self.dist_dir.exists():
                for dist_file in self.dist_dir.iterdir():
                    print(f"   - {dist_file.name}")
        else:
            print("❌ Multi-platform build failed!")
        
        return success

def main():
    """Main entry point."""
    builder = MultiPlatformBuilder()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--clean-only":
        builder.clean_build_dirs()
        print("✅ Clean completed")
        return
    
    success = builder.build_all()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 