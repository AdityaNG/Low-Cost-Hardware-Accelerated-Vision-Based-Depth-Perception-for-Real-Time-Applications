#include <GL/freeglut.h>
#include <GL/gl.h>
#include <stdlib.h>
#include <math.h>
#include <thread>

#define VEL_R 1
#define VEL_T 0.1

#define MOUSE_LEFT 			0
#define	MOUSE_MIDDLE 		1
#define	MOUSE_RIGHT 		2
#define	MOUSE_SCROLL_UP 	3
#define	MOUSE_SCROLL_DOWN	4

#define TRANSLATION_SENSITIVITY 0.01
#define GL_GLEXT_PROTOTYPES
#define SIZE 1

extern bool graphicsThreadExit;
extern bool draw_points; // Flag to enable or disable point cloud plotting
extern int out_width;
extern int out_height;
extern int Oindex;
extern int point_cloud_width;
extern int point_cloud_height;

template <typename P, typename C>
class Grapher {
    public:
        inline static P *points = NULL;
        inline static C *colors = NULL;
        inline static double *POINTS_OBJECTS = (double*) malloc(sizeof(double) * 9 * 50); // TODO: Remove hardcoding
        inline static double rX = 17; // Rotate X
        inline static double rY = 0;  // Rotate Y
        inline static double tX = 0; 
        inline static double tY = 0;
        inline static double tZ = 0;
        inline static double ZOOM = -0.2;

        Grapher() {
            points = (P*)malloc(sizeof(P) * point_cloud_width * point_cloud_height);
        }
        Grapher(P *p) { points = p; }

        P* getPointsArray() { return points; }
        C* getColorsArray() { return colors; }

        void setPointsArray(P *p) { points = p; }
        void setColorsArray(C *c) { colors = c; }

        void appendOBJECTS(double X, double Y, double Z, double r, double g, double b) {
            POINTS_OBJECTS[Oindex + 0] = X;
            POINTS_OBJECTS[Oindex + 1] = Y;
            POINTS_OBJECTS[Oindex + 2] = Z;
            POINTS_OBJECTS[Oindex + 3] = r;
            POINTS_OBJECTS[Oindex + 4] = g;
            POINTS_OBJECTS[Oindex + 5] = b;
            Oindex += 6;
        }
        
        static void draw_cube(double x, double y, double z, double r, double g, double b) {
            //printf("(%lf %lf %lf), ", x,y,z);
            double verts[8][3];
            for (int i = 0; i < 8; i++) {
                int s3 = ((1<<0) & i)>>0 ? 1: -1;
                int s2 = ((1<<1) & i)>>1 ? 1: -1;
                int s1 = ((1<<2) & i)>>2 ? 1: -1;
            
                verts[i][0] = x+ s1*SIZE/2.0;
                verts[i][1] = y+ s2*SIZE/2.0;
                verts[i][2] = z+ s3*SIZE/2.0;
            }
            glBegin(GL_LINE_STRIP);
            glColor3f(r, b, g);
            for (int iItr=0; iItr < 8; iItr++ ) {
                int i = iItr%8;
                for (int jItr = 0; jItr < 8; jItr++) {
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

        static void drawCube() {
            // Set Background Color
            //glClearColor(0.4, 0.4, 0.4, 1.0);
            glClearColor(0,0,0, 1.0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear screen

            glLoadIdentity(); // Reset transformations

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

            if(draw_points) {
                glPointSize(1);
                glBegin(GL_POINTS);
                for (int i = 0; i < out_width * out_height; i++) {
                    if(points == NULL) break;
                    if(colors == NULL) glColor3f(1.0, 1.0, 1.0);
                    else glColor3f(colors[i].x/255.0, colors[i].y/255.0, colors[i].z/255.0);
                    glVertex3f(points[i].x, points[i].y, points[i].z);
                }
                glEnd();
            }
            // (x, y, z) -> (-y, -z, x)
            int draw_radius = 1;
            if (draw_radius) {
                /*
                    x - Positive, along left of car
                    y - Positive, below ground
                    z - Positive, along direction of car

                    (x, y, z)
                    (z, y, x)
                    (z, x, y)
                    (z, -x, -y)
                */
                glPointSize(10);
                glBegin(GL_POINTS);
                glColor3f(0.0, 1.0, 0.0);
                glVertex3f(0, 0, 1);
                glEnd();   


                glPointSize(3);
                glBegin(GL_POINTS);
                for (int r = 1; r < 10; r++) {
                    /*for (float theta = 1.0; theta < 2*M_PI; theta+=2*M_PI/100) {
                        glColor3f(1.0, 0.0, 0.0);
                        glVertex3f(r * sin(theta), 0.0 , r * cos(theta));
                        // (x, y, z) -> (y, -z, x)
                    }*/

                    for (float theta = 0.0; theta < 2*M_PI; theta+=M_PI/100) {
                        glColor3f(1.0, 0.0, 0.0);
                        glVertex3f(-r * sin(theta), 0.0 , r * cos(theta));
                        
                    }
                }
                glEnd();  
            }
            
            
            draw_cube(0,0,0,1,0,0);
            for (int iObj = 0; iObj < Oindex; iObj+=6) {
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

        static void keyboard(int key, int x, int y) {
                 if (key == GLUT_KEY_RIGHT) rY += VEL_R;
            else if (key == GLUT_KEY_LEFT)  rY -= VEL_R;
            else if (key == GLUT_KEY_DOWN)  rX -= VEL_R;
            else if (key == GLUT_KEY_UP)    rX += VEL_R;

            // Request display update
            glutPostRedisplay();
        }

        static void keyboard_chars(unsigned char key, int x, int y) {
            switch(key) {
                case 'w': tY += VEL_T; break; 
                case 'a': tX -= VEL_T; break;
                case 's': tY -= VEL_T; break;
                case 'd': tX += VEL_T; break; 
                case 'e': ZOOM -= ZOOM * 0.2; break;
                case 'r': ZOOM += ZOOM * 0.2; break; 
                default: break;
            }
            glutPostRedisplay(); // Request display update
        }

        static void mouse_callback(int button, int state, int x, int y) {
            int xp=0, yp=0;
            //printf("%d %d %d %d\n", button, state, x, y);
            switch (button) {
                case MOUSE_LEFT:
                    if (state == 0) {
                        xp = x;
                        yp = y;
                    } 
                    else if (state == 1) {
                        rX = x-xp;
                        rY = y-yp;
                    }
                    break;

                case MOUSE_MIDDLE:
                    if (state == 0) {
                        xp = x;
                        yp = y;
                    } 
                    else if (state == 1) {
                        tX = (x-xp) * TRANSLATION_SENSITIVITY;
                        tY = -(y-yp) * TRANSLATION_SENSITIVITY;
                    } 
                    break;

                case MOUSE_SCROLL_UP:
                    if (state == 0) ZOOM += VEL_T;
                    break;

                case MOUSE_SCROLL_DOWN:
                    if (state == 0) ZOOM -= VEL_T;
                    break;

                default: break;
            }
            //printf("%f %f %f %f %f %f\n", ZOOM, rX, rY, tX, tY, tZ);
            // Request display update
            glutPostRedisplay();
        }

        void startGraphics() {            
            // Initialize GLUT and process user parameters
            int argc = 1;
            char *argv[1] = {(char*)"Something"};
            glutInit(&argc, argv);
            glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

            // Request double buffered true color window with Z-buffer
            glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);

            glutInitWindowSize(1400, 800);
            glutInitWindowPosition(100, 100);

            // Create window
            glutCreateWindow("3D depth map");

            // Enable Z-buffer depth test
            //glEnable(GL_DEPTH_TEST);
            //glDisable(GL_DEPTH_TEST);
            glEnable(GL_DEPTH_CLAMP);

            glutDisplayFunc(&Grapher::drawCube);
            glutSpecialFunc(&Grapher::keyboard);
            glutKeyboardFunc(&Grapher::keyboard_chars);
            glutMouseFunc(&Grapher::mouse_callback);
            glutIdleFunc(glutPostRedisplay);
            
            // Pass control to GLUT for events
            glutMainLoop();
            
            free(POINTS_OBJECTS);
            graphicsThreadExit = true;
        }
};