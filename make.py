#!/bin/python3
import os
import subprocess as proc
import argparse

parser = argparse.ArgumentParser("moonc")
parser.add_argument("-g", "--gdb", action="store_true", help="invoke gdb")
parser.add_argument("-c", "--cmake", action="store_true", help="re-run cmake")

args = parser.parse_args()

GDB = "gdb --args " if args.gdb else ""

os.chdir("build")

if args.cmake:
  proc.run("cmake -G Ninja -DCMAKE_CXX_COMPILER=clang++ ..", shell=True)

proc.run("ninja -j6", shell=True)
proc.run(f"{GDB}lib/main/moonc ../test/basic.mbt", shell=True)
