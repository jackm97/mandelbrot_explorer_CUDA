#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime_api.h>
#include <string>
#include <iostream>
#include <cmath>
#include <limits>
#include <string>
#include "mandelbrot.h"
#include "Shader.hpp"

using namespace std;

// The following functions handle getting input from the user
// for the mandelbrot image/animation parameters
void getResolution(int resolution[]);
void getCenter(string center[]);
void getSupersample(char supersample[]);
void getColorMap(string colorMap[]);
void getMaxIter(int max_iter[]);
void getZoom(float zoom[]);
void getZoomRange(float zoom_range[]);
void getFrameCount(int frame_count[]);
void getImagePath(string image_path[]);

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

// Checks user input to ensure correct program behavior.
// The string, prompt_str, is shown on the standard output
// until a valid input is given by the user
template<class var_type>
void input_check(string prompt_str, int argc, var_type argv[]);

bool MOVE_TEXTURE = false;
bool PRINT_LOCATION = false;
int DIRECTION = 0;
float ZOOM = 0;

int main(int argc, char *argv[]){

	string USAGE = \
		       "USAGE: ./mandelbrot_explorer arg1\n   arg1:  \"0\" for single image, \"1\" for zoom animation\n";
			
	if (argc!=2){
		cerr << USAGE;
		exit(1);
	}
	if (strcmp(argv[1],"0")!=0 && strcmp(argv[1],"1")!=0){
	       cerr << USAGE;	
	       exit(1);
	}

	int resolution[2];
	string center[2];
	char supersample[1];
        float zoom[1];
        float zoom_range[2];
        int max_iter[1], frame_count[1];
        string colorMap[1];
        //string image_path[1];
        
	getResolution(resolution);
	getCenter(center);
	getSupersample(supersample);
        getColorMap(colorMap);

        // If the user wants a single image	
	if (strcmp(argv[1],"0") == 0){
		getZoom(zoom);
		getMaxIter(max_iter);
        }
        // If the user wants a series of images to create a zoom animation
	else if(strcmp(argv[1],"1") == 0){
                getZoomRange(zoom_range);
                getFrameCount(frame_count);
                getMaxIter(max_iter);
                //getImagePath(image_path);
                zoom[0] = zoom_range[0];
        }

        int resScale = 1;
        if (supersample[0]=='y' || supersample[0]=='Y')
                resScale = 2;

        mandelbrot m(resolution[0] * resScale, resolution[1] * resScale, center, zoom[0], max_iter[0]);

        ZOOM = zoom[0];

        // glfw: initialize and configure
        // ------------------------------
        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        #ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
        #endif

        // glfw window creation
        // --------------------
        GLFWwindow* window = glfwCreateWindow(resolution[1], resolution[0], "Mandelbrot Explorer", NULL, NULL);
        if (window == NULL)
        {
                std::cout << "Failed to create GLFW window" << std::endl;
                glfwTerminate();
                return -1;
        }
        glfwMakeContextCurrent(window);
        glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
        glfwSetKeyCallback(window, key_callback);

        // glad: load all OpenGL function pointers
        // ---------------------------------------
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        {
                std::cout << "Failed to initialize GLAD" << std::endl;
                return -1;
        }

        // build and compile our shader zprogram
        // ------------------------------------
        string shaderPath = string("../src/colormap-shaders-master/shaders/glsl/") + colorMap[0] + string(".frag");
        Shader ourShader("../src/mandelbrot.vs", "../src/mandelbrot.fs", shaderPath.c_str());

        // vertices
        float vertices[] = {
        // positions            // texture coords
        1.0f,  1.0f, 0.0f,   1.0f, 1.0f,   // top right
        1.0f, -1.0f, 0.0f,   1.0f, 0.0f,   // bottom right
        -1.0f, -1.0f, 0.0f,   0.0f, 0.0f,   // bottom left
        -1.0f,  1.0f, 0.0f,   0.0f, 1.0f    // top left 
        };
        unsigned int indices[] = {
                0, 1, 3, // first triangle
                1, 2, 3  // second triangle
        };
        unsigned int VBO, VAO, EBO;
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

        // position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        
        // texture coord attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);   

        // generate texture:
        // ------------------
        GLuint texture;
        glGenTextures( 1, &texture );
        glBindTexture( GL_TEXTURE_2D, texture );
        
        // set basic parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        
        // Create texture data
        glTexImage2D( GL_TEXTURE_2D, 0, GL_R32F, resolution[1] * resScale, resolution[0] * resScale, 0, GL_RED, GL_FLOAT, NULL );

        m.registerTexture(texture);

        // Unbind the texture
        glBindTexture( GL_TEXTURE_2D, 0 );

	// If the user wants a single image	
	if (strcmp(argv[1],"0") == 0){

                // render loop
                // -----------
                while (!glfwWindowShouldClose(window))
                {
                        if (ZOOM!=zoom[0]){
                                m.changeZoom(ZOOM);
                                zoom[0] = ZOOM;
                        }

                        if (PRINT_LOCATION==true){
                                m.printLocation();
                                PRINT_LOCATION=false;
                        }

                        m.getImage();

                        if (MOVE_TEXTURE){
                                for (int i = 0; i<5; i++) m.moveDirection(DIRECTION);
                                MOVE_TEXTURE = false;
                        }

                        // input
                        // -----
                        processInput(window);

                        // render
                        // ------
                        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
                        glClear(GL_COLOR_BUFFER_BIT);

                        // bind textures on corresponding texture units
                        glActiveTexture(GL_TEXTURE0);
                        glBindTexture(GL_TEXTURE_2D, texture);

                        // render container
                        ourShader.use();
                        glBindVertexArray(VAO);
                        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

                        glfwSwapInterval(1);

                        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
                        // -------------------------------------------------------------------------------
                        glfwSwapBuffers(window);
                        glfwPollEvents();

                        // Unbind the texture
                        glBindTexture( GL_TEXTURE_2D, 0 );
                }

                // glfw: terminate, clearing all previously allocated GLFW resources.
                // ------------------------------------------------------------------
                glfwTerminate();
                return 0;
	}

	
	// If the user wants a series of images to create a zoom animation
	else if(strcmp(argv[1],"1") == 0){
		
                float zoom_min = zoom_range[0],
                zoom_max = zoom_range[1],
                zoom_interval;
		zoom_interval = ((zoom_max) - (zoom_min))/(frame_count[0]);

		while (!glfwWindowShouldClose(window)){
			m.changeZoom(zoom[0]);
                        
			m.getImage();

                        // input
                        // -----
                        processInput(window);

                        // render
                        // ------
                        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
                        glClear(GL_COLOR_BUFFER_BIT);

                        // bind textures on corresponding texture units
                        glActiveTexture(GL_TEXTURE0);
                        glBindTexture(GL_TEXTURE_2D, texture);

                        // render container
                        ourShader.use();
                        glBindVertexArray(VAO);
                        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

                        glfwSwapInterval(1);

                        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
                        // -------------------------------------------------------------------------------
                        glfwSwapBuffers(window);
                        glfwPollEvents();

                        // Unbind the texture
                        glBindTexture( GL_TEXTURE_2D, 0 );

			
			zoom[0] += zoom_interval;
                        if (zoom[0] >= zoom_max)
                                zoom[0] = zoom_min;
		}

                // glfw: terminate, clearing all previously allocated GLFW resources.
                // ------------------------------------------------------------------
                glfwTerminate();
                return 0;
	}

	return 0;
}

void getResolution(int resolution[]){
        cout << "Enter resolution, positive integers (e.g. height width): ";
        cin >> resolution[0] >> resolution[1];
        input_check("Enter resolution, positive integers (e.g. height width): ", 2, resolution);
        while (resolution[0] <= 0 || resolution[1] <= 0){
                cout << "You have entered the wrong input" << endl;
                cout << "Enter resolution, positive integers (e.g. height width): ";
                cin >> resolution[0] >> resolution[1];
                input_check("Enter resolution, positive integers (e.g. height width): ", 2, resolution);
        }
}

void getCenter(string center[]){
        char input[100];
        cin.getline(input,sizeof(input));
        cout << endl << "Enter x val: ";
        cin.getline(input,sizeof(input));
        center[0] = input;

        cout << endl << "Enter y val: ";
        cin.getline(input,sizeof(input));
        center[1] = input;
        
        //input_check("Enter center point (e.g. x y): ", 2, center);
}

void getSupersample(char supersample[]){
        cout << endl << "Supersample? (Y/n): ";
        cin >> supersample[0];
        input_check("Supersample? (Y/n): ", 1, supersample);
        while (supersample[0]!='y' && supersample[0]!='Y' && supersample[0]!='n' && supersample[0]!='N'){
                cout << "You have entered the wrong input" << endl;
                cout << "Supersample? (Y/n): ";
                cin >> supersample;
                input_check("Supersample? (Y/n): ", 1, supersample);
        }
}

void getColorMap(string colorMap[]){
        cout << endl << "Which colormap would you like to use (empty is default)?";
        cin >> colorMap[0];
        input_check("Which colormap would you like to use (empty is default)?", 1, colorMap);
        if (colorMap[0].length() == 0)
                colorMap[0] = "Default";
}

void getMaxIter(int max_iter[]){
                cout << endl << "Enter iterations(positive integer): ";
                cin >> max_iter[0];
                input_check("Enter iterations(positive integer): ", 1, max_iter);
                while (max_iter[0] <= 0){
                        cout << "You have entered the wrong input" << endl;
                        cout << "Enter iterations(positive integer): ";
                        cin >> max_iter[0];
                        input_check("Enter iterations(positive integer): ", 1, max_iter);
                }
}

void getZoom(float zoom[]){
                cout << endl <<  "Enter zoom level(maximum of 1e12): ";
                cin >> zoom[0];
                input_check("Enter zoom level(maximum of 1e12): ", 1, zoom);
                /*while (zoom[0] > 1e12 || zoom[0] <= 0){
                        cout << "You have entered the wrong input" << endl;
                        cout << "Enter zoom level(maximum of 1e12): ";
                        cin >> zoom[0];
                        input_check("Enter zoom level(maximum of 1e12): ", 1, zoom);
                }*/
}

void getZoomRange(float zoom_range[]){
                cout << endl << "Enter initial and final zoom level, maximum zoom is 1e12 (e.g. start_zoom end_zoom): ";
                cin >> zoom_range[0] >> zoom_range[1];
                input_check("Enter initial and final zoom level, maximum zoom is 1e12 (e.g. start_zoom end_zoom): ", 2, zoom_range);
                /*while (zoom_range[0] > 1e12 || zoom_range[0] <= 0 || zoom_range[1] > 1e12 || zoom_range[1] <= 0){
                        cout << "You have entered the wrong input" << endl;
                        cout << "Enter initial and final zoom level, maximum zoom is 1e12 (e.g. start_zoom end_zoom): ";
                        cin >> zoom_range[0] >> zoom_range[1];
                        input_check("Enter initial and final zoom level, maximum zoom is 1e12 (e.g. start_zoom end_zoom): ", 2, zoom_range);
                }*/
}

void getFrameCount(int frame_count[]){
                cout << endl << "Enter number of frames to capture (positive integer): ";
                cin >> frame_count[0];
                input_check("Enter number of frames to capture (positive integer): ", 1, frame_count);
                while (frame_count[0] <= 0 || fmod(frame_count[0],int(frame_count[0]))!=0){
                        cout << "You have entered the wrong input" << endl;
                        cout << "Enter number of frames to capture (positive integer): ";
                        cin >> frame_count[0];
                        input_check("Enter number of frames to capture (positive integer): ", 1, frame_count);
                }
}

void getImagePath(string image_path[]){
                cout << endl << "Enter file path to save image series: ";
                cin >> image_path[0];
                input_check("Enter file path to save image series: ", 1, image_path);
                if ( image_path[0].back()!='/' && image_path[0].length()!=0 )
                        image_path[0] += "/";
}


template <class var_type>
void input_check(string prompt_str, int argc, var_type argv[]){
	while (1){
		if (cin.fail()){
			cin.clear();
			cin.ignore(numeric_limits<streamsize>::max(),'\n');
			cout << "You have entered the wrong input" << endl;
			cout << prompt_str;
			for (int i=0; i<argc; i++)
				cin >> argv[i];
		}
		else
			return;
	}
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
        if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
                glfwSetWindowShouldClose(window, true);
        
        else if(glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS){
                DIRECTION = 0;
                MOVE_TEXTURE = true;
        }
        else if(glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS){
                DIRECTION = 1;
                MOVE_TEXTURE = true;
        }
        else if(glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS){
                DIRECTION = 2;
                MOVE_TEXTURE = true;
        }
        else if(glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS){
                DIRECTION = 3;
                MOVE_TEXTURE = true;
        }
        else if(glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS){
                ZOOM += .005;
        }
        else if(glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS){
                ZOOM -= .005;
        }
}
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
        if(key == GLFW_KEY_RIGHT && action == GLFW_PRESS){
                ZOOM += 1;
        }
        else if(key == GLFW_KEY_LEFT && action == GLFW_PRESS){
                ZOOM -= 1;
        }
        else if(key == GLFW_KEY_ENTER && action == GLFW_PRESS){
                PRINT_LOCATION = true;
        }
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}
