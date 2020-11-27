#include "graphing.h"
#include "../cleanup/cleanup.hpp"
#define GL_GLEXT_PROTOTYPES
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include <stdlib.h>
#include <iostream>
double *POINTS_OBJECTS;

// Rotate X
double rX=17;
// Rotate Y
double rY=0;

double tX=0, tY=-7., tZ=0, ZOOM=-0.2;

#define SIZE 1

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
    for (int i = 0; i < 8; i++){
        int s3 = ((1<<0) & i)>>0 ? 1: -1;
        int s2 = ((1<<1) & i)>>1 ? 1: -1;
        int s1 = ((1<<2) & i)>>2 ? 1: -1;
    
        verts[i][0] = x+ s1*SIZE/2.0;
        verts[i][1] = y+ s2*SIZE/2.0;
        verts[i][2] = z+ s3*SIZE/2.0;
    }

    glBegin(GL_LINE_STRIP);
    glColor3f(r, b, g);
    for (int iItr=0; iItr < 8; iItr++ ){
        int i = iItr%8;
        for (int jItr = 0; jItr < 8; jItr++){
            int j = jItr%8;
            glVertex3f(verts[i][0], verts[i][1], verts[i][2]);
            glVertex3f(verts[j][0], verts[j][1], verts[j][2]);
        }
    }
    glEnd();
  /*
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
  */
}
extern int out_width;
extern int out_height;
extern double3 *points;
extern uchar4 *color;
void drawCube(){
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

    int plane_size = 20;
    glBegin(GL_QUADS);
    glColor3f(0.5, 0.5, 0.5);
    glVertex3f(plane_size, 0, plane_size);
    glVertex3f(plane_size, 0, -plane_size);
    glVertex3f(-plane_size, 0, -plane_size);
    glVertex3f(-plane_size, 0, plane_size);
    glEnd();

    // BACK
    glBegin(GL_POINTS);
    for (int i = 0; i < out_width * out_height; i++){
        if(color == NULL) break;
		glColor3f(color[i].x/255.0, color[i].y/255.0, color[i].z/255.0);
		glPointSize(1);
		glVertex3f(points[i].y, -points[i].z, points[i].x);
    }
    glEnd();

    draw_cube(0,0,0, 1,0,0);
    for (int iObj = 0; iObj < Oindex; iObj+=6){
      draw_cube(POINTS_OBJECTS[iObj + 0], POINTS_OBJECTS[iObj + 1], POINTS_OBJECTS[iObj + 2], POINTS_OBJECTS[iObj + 3], POINTS_OBJECTS[iObj + 4], POINTS_OBJECTS[iObj + 5]);
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

void keyboard(int key, int x, int y){
         if (key == GLUT_KEY_RIGHT) rY += VEL_R;
    else if (key == GLUT_KEY_LEFT)  rY -= VEL_R;
    else if (key == GLUT_KEY_DOWN)  rX -= VEL_R;
    else if (key == GLUT_KEY_UP)    rX += VEL_R;

    // Request display update
    glutPostRedisplay();
}

void (*nextCALLBACK)(void);

void setCallback(void (*f)(void)) {
    nextCALLBACK = f;
}

void keyboard_chars(unsigned char key, int x, int y)
{
         if (key == 'w') tY += VEL_T;
    else if (key == 'a') tX -= VEL_T;
    else if (key == 's') tY -= VEL_T;
    else if (key == 'd') tX += VEL_T; 
    else if (key == 'q') clean();
    else if (key == 'e') ZOOM -= ZOOM * 0.2;//VEL_T; 
    else if (key == 'r') ZOOM +=  ZOOM * 0.2;//VEL_T; 
    else if (key == 'n'){
        if (nextCALLBACK) nextCALLBACK();
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

	if (button == MOUSE_LEFT){
		if (state == 0){
			xp = x;
			yp = y;
		} 
        else if (state == 1){
			rX = x-xp;
			rY = y-yp;
		}
	} 
    else if (button == MOUSE_MIDDLE){
		if (state == 0){
			xp = x;
			yp = y;
		} 
        else if (state == 1){
			tX = (x-xp) * TRANSLATION_SENSITIVITY;
			tY = -(y-yp) * TRANSLATION_SENSITIVITY;
		}
	} 
    else if (button == MOUSE_SCROLL_UP){
		if (state == 0){
			ZOOM += VEL_T;
		}
	} 
    else if (button == MOUSE_SCROLL_DOWN){
		if (state == 0){
			ZOOM -= VEL_T;
		}
	}
    //printf("%f %f %f %f %f %f\n", ZOOM, rX, rY, tX, tY, tZ);
    // Request display update
    glutPostRedisplay();
}

void startGraphics(int out_width, int out_height) {
    POINTS_OBJECTS = (double*) malloc(sizeof(double) * 9 * 50);
	
    // Initialize GLUT and process user parameters
    int argc = 1;
    char *argv[1] = {(char*)"Something"};
    glutInit(&argc, argv);

    // Request double buffered true color window with Z-buffer
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);

    glutInitWindowSize(1400,800);
    glutInitWindowPosition(100, 100);

    // Create window
    glutCreateWindow("3D depth map");

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
    free(POINTS_OBJECTS);
}