

workspace "LearnVulkan"
configurations { "Debug", "Release" }
platforms { "x64" }
location "./build"
filename ("LearnVulkan")
startproject "LearnVulkan"

filter "platforms:x64"
system "Windows"
architecture "x64"

project "LearnVulkan"
kind "ConsoleApp"
language "C++"
cppdialect "C++17"
location "./build"
warnings "Extra" -- /W4
filename ("LearnVulkan")
targetdir "./build/bin"
prebuildcommands {
  "%{cfg.basedir}/compile_shader.bat"
}
objdir "./build/%{cfg.platform}/%{cfg.buildcfg}"
files { "./Sources/*.h", "./Sources/*.cpp" }
includedirs { "./ThirdParty/glfw-3.3.8.bin.WIN64/include" }
includedirs { "./ThirdParty/glm" }
includedirs { "./ThirdParty/VulkanSDK/include" }
includedirs { "./ThirdParty/tinyobjloader-release" }
includedirs { "./ThirdParty/stb-master" }
libdirs{"./ThirdParty/glfw-3.3.8.bin.WIN64/lib-vc2022", "./ThirdParty/VulkanSDK/lib/Win32"}
links{"glfw3", "vulkan-1"}
characterset "Unicode"

filter "configurations:Debug"
defines { "_DEBUG", "DEBUG" }
targetsuffix ("_Debug")

filter "configurations:Release"
defines { "NDEBUG" }
optimize "On"
flags { "LinkTimeOptimization" }
targetsuffix ("_Release")

filter { "configurations:Debug", "platforms:x64" }
buildoptions { "/MDd" }

filter { "configurations:Release", "platforms:Windows-x64" }
buildoptions { "/MD" }