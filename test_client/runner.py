#!/usr/bin/env python3
import asyncio
from scripts import test_timing

# Comment out tests you don't want to run
async def main():
    await test_timing.main(
        url="http://host.docker.internal:8880",
        voice="af_bella",
        speed=1.0,
        text="The quick brown fox jumps over the lazy dog."
    )

if __name__ == "__main__":
    asyncio.run(main())