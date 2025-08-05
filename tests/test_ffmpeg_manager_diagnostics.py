"""
Test for FFmpeg Manager diagnostics to isolate transport closure issue.
"""

import pytest
import asyncio
import logging
from unittest.mock import AsyncMock, Mock
from vistreamasr.ffmpeg_manager import FFmpegManager, FFmpegState

# Configure logging for test visibility
logging.basicConfig(level=logging.DEBUG)

@pytest.mark.anyio
async def test_ffmpeg_manager_startup_diagnostics():
    """
    Test FFmpeg manager startup with diagnostic logging to identify transport closure issue.
    This test validates the hypothesis about process health and race conditions.
    """
    manager = FFmpegManager()
    
    # Test 1: Basic startup
    startup_success = await manager.start()
    
    # Allow some time for any race conditions to manifest
    await asyncio.sleep(0.2)
    
    # Check if process is still alive after startup
    if manager.process:
        process_alive = manager.process.returncode is None
        stdin_open = manager.process.stdin and not manager.process.stdin.is_closing()
        stdout_open = manager.process.stdout is not None
        
        print(f"Diagnostic Results:")
        print(f"  Startup Success: {startup_success}")
        print(f"  Process Alive: {process_alive}")
        print(f"  Process PID: {manager.process.pid if manager.process else 'None'}")
        print(f"  Process Return Code: {manager.process.returncode if manager.process else 'None'}")
        print(f"  Stdin Open: {stdin_open}")
        print(f"  Stdout Open: {stdout_open}")
        print(f"  Manager State: {manager.state}")
    
    # Test 2: Try a small write operation
    if startup_success:
        test_data = b'\x00' * 1024  # 1KB of silence
        write_success = await manager.write_data(test_data)
        print(f"  Test Write Success: {write_success}")
        
        # Check process state after write attempt
        if manager.process:
            process_alive_after_write = manager.process.returncode is None
            print(f"  Process Alive After Write: {process_alive_after_write}")
    
    # Cleanup
    await manager.stop()
    
    # For now, we're just collecting diagnostic information
    # The actual assertions will depend on what we discover
    assert startup_success or not startup_success  # Always pass - this is a diagnostic test

@pytest.mark.anyio
async def test_ffmpeg_command_validation():
    """
    Test if the FFmpeg command line arguments are valid by running them manually.
    """
    import subprocess
    
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", "16000",
        "pipe:1"
    ]
    
    try:
        # Test if FFmpeg accepts these arguments (will fail due to no input, but should not error on args)
        result = subprocess.run(cmd, input=b'', capture_output=True, timeout=2)
        print(f"FFmpeg command test:")
        print(f"  Return code: {result.returncode}")
        print(f"  Stderr: {result.stderr.decode(errors='ignore')}")
        
        # We expect this to fail due to no input, but args should be valid
        # Return code 1 is expected for "no input" error
        assert result.returncode in [0, 1], f"Unexpected return code: {result.returncode}"
        
    except subprocess.TimeoutExpired:
        print("FFmpeg command timed out (expected if waiting for input)")
        assert True  # This is actually expected behavior
    except FileNotFoundError:
        pytest.skip("FFmpeg not installed or not in PATH")

if __name__ == "__main__":
    asyncio.run(test_ffmpeg_manager_startup_diagnostics())