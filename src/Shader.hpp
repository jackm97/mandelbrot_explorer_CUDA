#ifndef SHADER_HPP
#define SHADER_HPP

#include <string>

class Shader
{
public:
	// program ID
	unsigned int ID;

	// Constructor reads and builds shader
	Shader(const char* vertexPath, const char* fragmentPath);
	// activate shader
	void use();
	// set uniform functions
	void setBool(const std::string &name, bool value) const;
	void setInt(const std::string &name, int value) const;
	void setFloat(const std::string &name, float value) const;
};

#endif
