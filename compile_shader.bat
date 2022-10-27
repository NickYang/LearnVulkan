echo "complie shader to Resources"
echo %~dp0
%~dp0ThirdParty/VulkanSDK/bin/Win32/glslc.exe %~dp0Shaders/Source/hello_triangle.vert -o %~dp0Resources/Shaders/hello_triangle_vert.spv
%~dp0ThirdParty/VulkanSDK/bin/Win32/glslc.exe %~dp0Shaders/Source/hello_triangle.frag -o %~dp0Resources/Shaders/hello_triangle_frag.spv
echo "finished compile shader to Resources"
