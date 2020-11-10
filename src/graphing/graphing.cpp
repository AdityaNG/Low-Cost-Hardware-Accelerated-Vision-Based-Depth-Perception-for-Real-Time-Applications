#include "graphing.h"
#define GL_GLEXT_PROTOTYPES
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include <stdlib.h>
#include <iostream>
float *POINTS, *POINTS_OBJECTS;

// Rotate X
double rX=17;
// Rotate Y
double rY=0;

double tX=0, tY=-7., tZ=0, ZOOM=-0.2;

#define SIZE 1

// The coordinates for the vertices of the cube

int Pindex = 0;
void appendPOINT(double X, double Y, double Z, double r, double g, double b) {
    POINTS[Pindex + 0] = X;
    POINTS[Pindex + 1] = Y;
    POINTS[Pindex + 2] = Z;
    POINTS[Pindex + 3] = r;
    POINTS[Pindex + 4] = g;
    POINTS[Pindex + 5] = b;
    Pindex+=6;
}
void resetPOINTS() {
    Pindex = 0;
}

int Oindex = 0;
void appendOBJECTS(double X, double Y, double Z, double r, double g, double b) {
    POINTS_OBJECTS[Oindex + 0] = X;
    POINTS_OBJECTS[Oindex + 1] = Y;
    POINTS_OBJECTS[Oindex + 2] = Z;
    POINTS_OBJECTS[Oindex + 3] = r;
    POINTS_OBJECTS[Oindex + 4] = g;
    POINTS_OBJECTS[Oindex + 5] = b;
    Oindex+=6;
}
void resetOBJECTS() {
    Oindex = 0;
}

void draw_cube(double x, double y, double z, double r, double g, double b) {

  //printf("(%lf %lf %lf), ", x,y,z);

  double verts[8][3];
  for (int i = 0; i < 8; i++)
  {
    int s3 = ((1<<0) & i)>>0 ? 1: -1;
    int s2 = ((1<<1) & i)>>1 ? 1: -1;
    int s1 = ((1<<2) & i)>>2 ? 1: -1;
    
    verts[i][0] = x+ s1*SIZE/2.0;
    verts[i][1] = y+ s2*SIZE/2.0;
    verts[i][2] = z+ s3*SIZE/2.0;
  }


  glBegin(GL_QUADS);
    glColor3f(r, b, g);
    for (int j = 0; j < 8; j++) {
      int i = j%8;
      glVertex3f(verts[i][0], verts[i][1], verts[i][2]);
      glVertex3f(verts[i+1][0], verts[i+1][1], verts[i+1][2]);
      glVertex3f(verts[i+2][0], verts[i+2][1], verts[i+2][2]);
      glVertex3f(verts[i+3][0], verts[i+3][1], verts[i+3][2]);
    }
  glEnd();
}

void drawCube()
{
        // Set Background Color
    //glClearColor(0.4, 0.4, 0.4, 1.0);
    glClearColor(0,0,0, 1.0);
        // Clear screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Reset transformations
    glLoadIdentity();

    // Rotate when user changes rX and rY
    glRotatef( rX, 1.0, 0.0, 0.0 );
    glRotatef( rY, 0.0, 1.0, 0.0 );


    glScalef(ZOOM, ZOOM, ZOOM);
    
    glTranslatef(tX, tZ, tY);


    // BACK
        glBegin(GL_POINTS);
        for (int i = 0; i < Pindex; i+=6)
				{
					glColor3f(POINTS[i+3], POINTS[i+4], POINTS[i+5]);
					glPointSize(1);
					glVertex3f(POINTS[i], POINTS[i+1], POINTS[i+2]);
				}
        glEnd();

    draw_cube(0,0,0, 0,1,0);
    for (int iObj = 0; iObj < Oindex; iObj+=6)
    {
      draw_cube(POINTS_OBJECTS[iObj + 0], POINTS_OBJECTS[iObj + 1], POINTS_OBJECTS[iObj + 2],1, 0,0);
      /*
        POINTS_OBJECTS[iObj + 0] = X;
        POINTS_OBJECTS[iObj + 1] = Y;
        POINTS_OBJECTS[iObj + 2] = Z;
        POINTS_OBJECTS[iObj + 3] = red    / 255.0;
        POINTS_OBJECTS[iObj + 4] = green  / 255.0;
        POINTS_OBJECTS[iObj + 5] = blue   / 255.0;
      */
    }
    

    glFlush();
    glutSwapBuffers();
}

void keyboard(int key, int x, int y)
{
    if (key == GLUT_KEY_RIGHT)
        {
                rY += VEL_R;
        }
    else if (key == GLUT_KEY_LEFT)
        {
                rY -= VEL_R;
        }
    else if (key == GLUT_KEY_DOWN)
        {
                rX -= VEL_R;
        }
    else if (key == GLUT_KEY_UP)
        {
                rX += VEL_R;
        }

    // Request display update
    glutPostRedisplay();
}

void (*nextCALLBACK)(void);

void setCallback(void (*f)(void)) {
    nextCALLBACK = f;
}

void keyboard_chars(unsigned char key, int x, int y)
{
    if (key == 'w')
        {
            tY += VEL_T;
        }
    else if (key == 'a')
        {
                tX -= VEL_T;
        }
    else if (key == 's')
        {
                tY -= VEL_T;
        }
    else if (key == 'd')
        {
                tX += VEL_T; 
        }
    else if (key == 'q')
        {
                exit(0);
        }
    else if (key == 'e')
        {
                ZOOM -= VEL_T; 
        }
    else if (key == 'r')
        {
                ZOOM += VEL_T; 
        }
    else if (key == 'n')
        {
            if (nextCALLBACK)
                nextCALLBACK();
        }
    // Request display update
    glutPostRedisplay();
}

void updateGraph() {
    glutPostRedisplay();
}


void mouse_callback(int button, int state, int x, int y) {
	static int xp=0, yp=0;
	//printf("%d %d %d %d\n", button, state, x, y);

	if (button == MOUSE_LEFT) {
		if (state == 0) {
			xp = x;
			yp = y;
		} else if (state == 1) {
			rX = x-xp;
			rY = y-yp;
		}
	} else if (button == MOUSE_MIDDLE) {
		if (state == 0) {
			xp = x;
			yp = y;
		} else if (state == 1) {
			tX = (x-xp) * TRANSLATION_SENSITIVITY;
			tY = -(y-yp) * TRANSLATION_SENSITIVITY;
		}
	} else if (button == MOUSE_SCROLL_UP) {
		if (state == 0) {
			ZOOM += VEL_T;
		}
	} else if (button == MOUSE_SCROLL_DOWN) {
		if (state == 0) {
			ZOOM -= VEL_T;
		}
	}
  printf("%f %f %f %f %f %f\n", ZOOM, rX, rY, tX, tY, tZ);
    // Request display update
    glutPostRedisplay();
}

void startGraphics(int out_width, int out_height) {
  POINTS = (float*) malloc(sizeof(float) * 6 * out_width * out_height);
  POINTS_OBJECTS = (float*) malloc(sizeof(float) * 9 * 50);
	
	for (int i = 0; i < 100; i+=6)
	{
		POINTS[i]	= i/400.0;
		POINTS[i+1]	= i/400.0;
		POINTS[i+2]	= i/400.0;
		POINTS[i+3]	= i/400 + 0.5;
		POINTS[i+4]	= i/400 + 0.5;
		POINTS[i+5]	= i/400 + 0.5;
	}
        // Initialize GLUT and process user parameters
        int argc = 1;
        char *argv[1] = {(char*)"Something"};
        glutInit(&argc, argv);

        // Request double buffered true color window with Z-buffer
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);

        glutInitWindowSize(1400,800);
        glutInitWindowPosition(100, 100);

        // Create window
        glutCreateWindow("Linux Journal OpenGL Cube");

        // Enable Z-buffer depth test
        //glEnable(GL_DEPTH_TEST);
        //glDisable(GL_DEPTH_TEST);
        glEnable(GL_DEPTH_CLAMP);

        // Callback functions
        glutDisplayFunc(drawCube);
        glutSpecialFunc(keyboard);
        glutKeyboardFunc(keyboard_chars);
        glutMouseFunc(mouse_callback);
		//thread th1(test); 
	
        // Pass control to GLUT for events
        glutMainLoop();
	
		//th1.join();

    free(POINTS);
    free(POINTS_OBJECTS);
}