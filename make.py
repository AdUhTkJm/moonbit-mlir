#!/bin/python3
import os
import subprocess as proc
import argparse

parser = argparse.ArgumentParser("moonc")
parser.add_argument("-g", "--gdb", action="store_true", help="invoke gdb")
parser.add_argument("-c", "--cmake", action="store_true", help="re-run cmake")
parser.add_argument("-a", "--ast", action="store_true", help="dump ast")
parser.add_argument("-i", "--ir", action="store_true", help="dump ir")
parser.add_argument("-t", "--test", type=str, help="run this test case")

args = parser.parse_args()

GDB = "gdb --args " if args.gdb else ""
DUMPAST = " -dump-ast" if args.ast else ""
DUMPIR = " -dump-ir" if args.ir else ""

os.chdir("build")

if args.cmake:
  proc.run("cmake -G Ninja -DCMAKE_CXX_COMPILER=clang++ ..", shell=True)

proc.run("ninja -j16", shell=True)
print("Done.")
if args.test:
  proc.run(f"{GDB}lib/main/moonc ../test/{args.test}{DUMPAST}{DUMPIR}", shell=True)
