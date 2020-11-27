#define VEL_R 1
#define VEL_T 0.1

#define MOUSE_LEFT 			0
#define	MOUSE_MIDDLE 		1
#define	MOUSE_RIGHT 		2
#define	MOUSE_SCROLL_UP 	3
#define	MOUSE_SCROLL_DOWN	4

#define TRANSLATION_SENSITIVITY 0.01

#define SIZE 1

void appendPOINT(double X, double Y, double Z, double r, double g, double b);
void resetPOINTS();

void appendOBJECTS(double X, double Y, double Z, double r, double g, double b);
void resetOBJECTS();

void updateGraph();
void setCallback(void (*f)(void));

void startGraphics(int out_width, int out_height);