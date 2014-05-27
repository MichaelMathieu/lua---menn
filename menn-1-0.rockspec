package = "menn"
version = "1-0"

source = {
   url = "git://github.com/MichaelMathieu/lua---menn.git",
}

description = {
   summary = "Memory Efficient Neural Network package for Torch",
   detailed = [[
   ]],
   homepage = "",
   license = ""
}

dependencies = {
   "torch >= 7.0",
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
]],
   install_command = "cd build && $(MAKE) install"
}
