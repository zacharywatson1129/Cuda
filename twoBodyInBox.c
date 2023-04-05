// gcc twoBodyInBox.c -o temp2 -lglut -lm -lGLU -lGL
//To stop hit "control c" in the window you launched it from.
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUMBER_OF_SPHERES 10

#define XWindowSize 1000
#define YWindowSize 1000

#define STOP_TIME 10000.0
#define DT        0.0001

#define GRAVITY 2.0

#define MASS 10.0  	
#define DIAMETER 1.0

#define SPRING_STRENGTH 50.0 // how hard the ball is. is it made of rubber or porcelain?
#define SPRING_REDUCTION 1.0

#define DAMP 0.1

#define DRAW 100

#define LENGTH_OF_BOX 6.0
#define MAX_VELOCITY 1.0

const float XMax = (LENGTH_OF_BOX/2.0);
const float YMax = (LENGTH_OF_BOX/2.0);
const float ZMax = (LENGTH_OF_BOX/2.0);
const float XMin = -(LENGTH_OF_BOX/2.0);
const float YMin = -(LENGTH_OF_BOX/2.0);
const float ZMin = -(LENGTH_OF_BOX/2.0);

// Globals
float *px, *py, *pz, *vx, *vy, *vz, *fx, *fy, *fz, *mass;

void allocateMemory()
{
	px = (float*) malloc(sizeof(float) * NUMBER_OF_SPHERES);
	py = (float*) malloc(sizeof(float) * NUMBER_OF_SPHERES);
	pz = (float*) malloc(sizeof(float) * NUMBER_OF_SPHERES);

	vx = (float*) malloc(sizeof(float) * NUMBER_OF_SPHERES);
	vy = (float*) malloc(sizeof(float) * NUMBER_OF_SPHERES);
	vz = (float*) malloc(sizeof(float) * NUMBER_OF_SPHERES);

	fx = (float*) malloc(sizeof(float) * NUMBER_OF_SPHERES);
	fy = (float*) malloc(sizeof(float) * NUMBER_OF_SPHERES);
	fz = (float*) malloc(sizeof(float) * NUMBER_OF_SPHERES);

	mass = (float*) malloc(sizeof(float) * NUMBER_OF_SPHERES);
}

void cleanup()
{
	free(px); free(py); free(pz);
	free(vx); free(vy); free(vz);
	free(fx); free(fy); free(fz);
	free(mass);
}

void set_initail_conditions()
{ 
	// Used for generating random numbers.<
	time_t t;
	srand((unsigned) time(&t));
	int yeahBuddy;
	float dx, dy, dz, seperation;
	// float tempPx, tempPy, tempPz;
	// int i = 0;

	allocateMemory();
	printf("\n Memory has been allocated \n");
	
	// Get a random number for the center of the first sphere.
	px[0] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
	py[0] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
	pz[0] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;

	vx[0] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
	vy[0] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
	vz[0] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
	
	mass[0] = 1.0;


	for (int i = 1; i < NUMBER_OF_SPHERES; i++)
	{
		// These variables don't matter the value, so they can be set to a random value and it's good.
		vx[i] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		vy[i] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		vz[i] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
	
		mass[i] = 1.0;

		yeahBuddy = 0;
		while(yeahBuddy == 0)
		{
			// Get a random number for components of coordinates of the ith sphere.
			px[i] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
			py[i] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
			pz[i] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
			
			// Now, it cannot intersect with every other sphere so far.
			for (int j = 0; j < i; j++)
			{
				dx = px[i] - px[j];
				dy = py[i] - py[j];
				dz = pz[i] - pz[j];
				seperation = sqrt(dx*dx + dy*dy + dz*dz);
				yeahBuddy = 1;
				if (seperation < DIAMETER) 
				{
					yeahBuddy = 0; // So that the while loop reruns.
					break;
				}
			}
		}
	}
}

void Drawwirebox()
{		
	glColor3f (5.0,1.0,1.0);
	glBegin(GL_LINE_STRIP);
		glVertex3f(XMax,YMax,ZMax);
		glVertex3f(XMax,YMax,ZMin);	
		glVertex3f(XMax,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMax);
		glVertex3f(XMax,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		glVertex3f(XMin,YMax,ZMin);	
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMin,YMax,ZMax);	
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMax,YMin,ZMax);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMin);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMax,ZMin);
		glVertex3f(XMax,YMax,ZMin);		
	glEnd();
	
}

void draw_picture()
{
	float radius = DIAMETER/2.0;
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	Drawwirebox();

	// This has to be replaced with a loop that draws each sphere.
	for (int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		glColor3d(1.0,0.5,1.0);
		glPushMatrix();
		glTranslatef(px[i], py[i], pz[i]);
		glutSolidSphere(radius,20,20);
		glPopMatrix();
	}
	
	glutSwapBuffers();
}

void keep_in_box()
{
	float halfBoxLength = (LENGTH_OF_BOX - DIAMETER)/2.0;
	
	for (int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		if(px[i] > halfBoxLength)
		{
			px[i] = 2.0*halfBoxLength - px[i];
			vx[i] = - vx[i];
		}
		else if(px[i] < -halfBoxLength)
		{
			px[i] = -2.0*halfBoxLength - px[i];
			vx[i] = - vx[i];
		}
		
		if(py[i] > halfBoxLength)
		{
			py[i] = 2.0*halfBoxLength - py[i];
			vy[i] = - vy[i];
		}
		else if(py[i] < -halfBoxLength)
		{
			py[i] = -2.0*halfBoxLength - py[i];
			vy[i] = - vy[i];
		}
				
		if(pz[i] > halfBoxLength)
		{
			pz[i] = 2.0*halfBoxLength - pz[i];
			vz[i] = - vz[i];
		}
		else if(pz[i] < -halfBoxLength)
		{
			pz[i] = -2.0*halfBoxLength - pz[i];
			vz[i] = - vz[i];
		}
	}
}

// This sort of works...objects don't collide, but it's not exactly how
// I want them to behave.
void handleCollisions()
{
	for (int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		for (int j = 0; j < NUMBER_OF_SPHERES; j++)
		{
			if (j == i)
				continue;
			float dx = px[i] - px[j];
			float dy = py[i] - py[j];
			float dz = pz[i] - pz[j];
			float d2 = dx*dx + dy*dy + dz*dz;
			float d = sqrt(d2);
			if (d <= DIAMETER) // if less than 2 diameters apart, that means they have collided 
			{
				// First adjust the positions so that we're not colliding anymore.
				// px[i] =2.0*halfBoxLength - px[i];

				// Now let's change the velocities.

				// -------- Option 1 ----------
				vx[i] = - vx[i];
				vy[i] = - vy[i];
				vz[i] = - vz[i];

				vx[j] = - vx[j];
				vy[j] = - vy[j];
				vz[j] = - vz[j];
				// -------- Option 2 -----------
				/*float temp = vx[i];
				vx[i] += vx[j];
				vx[j] = temp;

				temp = vy[i];
				vy[i] = vy[j];
				vy[j] = temp;

				temp = vz[i];
				vz[i] = vz[j];
				vz[j] = temp;*/
				// -----------------------------

				// We didn't fix positions, so I just go ahead and move it one time step forward.
				px[i] += DT*vx[i];
				py[i] += DT*vy[i];
				pz[i] += DT*vz[i];

				px[j] += DT*vx[j];
				py[j] += DT*vy[j];
				pz[j] += DT*vz[j];
			}
		}
	}
}

void get_forces()
{
	float dx,dy,dz,r,r2,dvx,dvy,dvz,forceMag,inout;
	
	for (int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		for (int j = 0; j < NUMBER_OF_SPHERES; j++)
		{
			if (j == i) // Don't calculate forces with yourself.
				continue;
			dx = px[j] - px[i];
			dy = py[j] - py[i];
			dz = pz[j] - pz[i];

			r2 = dx*dx + dy*dy + dz*dz;
			r = sqrt(r2);

			forceMag = mass[i]*mass[j]*GRAVITY/2;

			if (r < DIAMETER)
			{
				dvx = vx[j] - vx[i];
				dvy = vy[j] - vy[i];
				dvz = vz[j] - vz[i];
				inout = dx*dvx + dy*dvy + dz*dvz;
				if(inout <= 0.0)
				{
					forceMag +=  SPRING_STRENGTH*(r - DIAMETER);
				}
				else
				{
					forceMag +=  SPRING_REDUCTION*SPRING_STRENGTH*(r - DIAMETER);
				}
			}

			fx[i] = forceMag*dx/r;
			fy[i] = forceMag*dy/r;
			fz[i] = forceMag*dz/r;
			fx[j] = -forceMag*dx/r;
			fy[j] = -forceMag*dy/r;
			fz[j] = -forceMag*dz/r;
		}
	}
}

void move_bodies(float time)
{
	if(time == 0.0)
	{
		for (int i = 0; i < NUMBER_OF_SPHERES; i++)
		{
			vx[i] += 0.5*DT*(fx[i] - DAMP*vx[i])/mass[i];
			vy[i] += 0.5*DT*(fy[i] - DAMP*vy[i])/mass[i];
			vz[i] += 0.5*DT*(fz[i] - DAMP*vz[i])/mass[i];
		}
	}
	else
	{
		for (int i = 0; i < NUMBER_OF_SPHERES; i++)
		{
			vx[i] += DT*(fx[i] - DAMP*vx[i])/mass[i];
			vy[i] += DT*(fy[i] - DAMP*vy[i])/mass[i];
			vz[i] += DT*(fz[i] - DAMP*vz[i])/mass[i];
		}
	}

	for (int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		px[i] += DT*vx[i];
		py[i] += DT*vy[i];
		pz[i] += DT*vz[i];
	}
	
	keep_in_box();
	//handleCollisions();
}

void nbody()
{	
	int    tdraw = 0;
	float  time = 0.0;

	set_initail_conditions();
	draw_picture();
	
	while(time < STOP_TIME)
	{
		get_forces();
	
		move_bodies(time);
	
		tdraw++;
		if(tdraw == DRAW) 
		{
			draw_picture(); 
			tdraw = 0;
		}
		
		time += DT;
	}
	printf("\n DONE \n");
	while(1);
}

void Display(void)
{
	gluLookAt(0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glutSwapBuffers();
	glFlush();
	nbody();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);

	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();

	glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, 50.0);

	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv)
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("N Body 3D");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutMainLoop();
	cleanup();
	return 0;
}
