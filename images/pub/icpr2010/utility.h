////////////////////////////////////////////////////////////////////
// utility.h: computing local surface properties on range image
// Neslihan Bayramoglu
//               
// Copyright 2010 Neslihan Bayramoglu
//
// Reference:
// Bayramoglu, Neslihan, and A. Aydin Alatan. 
// "Shape index sift: Range image recognition using local features." 
// Pattern Recognition (ICPR), 2010 20th International Conference on. IEEE, 2010.
//
// Depends on the OpenCV library
//////////////////////////////////////////////////////////////////////

#include "math.h"
#include <cv.h>	
#include <cvaux.h>
#include <highgui.h>
#include <ml.h>

#include <vector>
using namespace std;



void findPolySurPatches(const int filterWidth, const int filterHeight, const CvMat* _X, const CvMat* _Y, const CvMat* _Z, CvMat* quadPatchParam )
{
	// given the range data information with _X,_Y, and _Z matrices (3D coordinates of each pixel), 
	// find the quadratic surface coefficients for a local patch of size (filterWidth x filterHeight)
	// that satisfies the following polynomial surface equation:
	// ax^2+by^2+cxy+dx+ey+f=z 
	// In order to perform mean normalization and stdandard deviation normalization modify the code by uncommenting the 
	// related lines

	int pointCount = filterWidth*filterHeight;
	int centerPointIndex = (filterWidth * filterHeight / 2);
	CvMat* patchMembers = cvCreateMat( pointCount, 3, CV_32FC1 );
	CvMat* A = cvCreateMat( 6, 6, CV_32FC1 );
	CvMat* W = cvCreateMat( 6, 6, CV_32FC1 );
	CvMat* U = cvCreateMat( 6, 6, CV_32FC1 );
	CvMat* V = cvCreateMat( 6, 6, CV_32FC1 );
	CvMat* D = cvCreateMat( pointCount, 6, CV_32FC1 );
	CvMat* DT = cvCreateMat( 6, pointCount, CV_32FC1 );
	CvMat* T = cvCreateMat(3,3,CV_32FC1);
	CvMat* WLS = cvCreateMat( pointCount, 1, CV_32FC1 );

	CvMat* estimate = cvCreateMat(_X->rows, _X->cols, CV_32FC1);
	CvMat* X = cvCreateMat(_X->rows, _X->cols, CV_32FC1);
	CvMat* Y = cvCreateMat(_X->rows, _X->cols, CV_32FC1);
	cvCopy(_X,X);
	cvCopy(_Y,Y);


	int halfFilterX = filterWidth /2;
	int halfFilterY = filterHeight/2;
	bool borderCOntrol = false;
	int minDepthInd = 0;
	float minDepth;

	
	for(int indY=0; indY<_X->rows; indY++)
		for(int indX=0; indX< _X->cols; indX++)
		{
			if (_X->data.fl[indY*_X->cols + indX] == 0 && _X->data.fl[indY*_X->cols + indX] == 0 && _Z->data.fl[indY*_X->cols + indX] == 0)
			{
				int index = indY*_X->cols + indX;

		
			for(int param = 0; param < 6; param++)
					quadPatchParam->data.fl[6*index + param] = 0 ;
			}
			else
		{
			minDepth = _Z->data.fl[indY*_X->cols + indX];
			int count = 0;
			borderCOntrol = false;
			for(int j=indY-halfFilterY; j<=indY+halfFilterY; j++)
				for(int i=indX-halfFilterX; i<=indX+halfFilterX; i++)
				{
					if(i<0 || i>=_X->cols || j<0 || j>=_X->rows )
					{
						patchMembers->data.fl[3*count  ] = _X->data.fl[indY*_X->cols + indX];
						patchMembers->data.fl[3*count+1] = _Y->data.fl[indY*_X->cols + indX];
						patchMembers->data.fl[3*count+2] = _Z->data.fl[indY*_X->cols + indX];	
						borderCOntrol = true;
						
					}
					else 
						if (_X->data.fl[j*_X->cols + i] == 0 && _X->data.fl[j*_X->cols + i] == 0 && _Z->data.fl[j*_X->cols + i] == 0)
		
						{
							patchMembers->data.fl[3*count  ] = _X->data.fl[indY*_X->cols + indX];
						patchMembers->data.fl[3*count+1] = _Y->data.fl[indY*_X->cols + indX];
						patchMembers->data.fl[3*count+2] = _Z->data.fl[indY*_X->cols + indX];	
						}
						else
					{
						patchMembers->data.fl[3*count  ] = _X->data.fl[j*_X->cols + i];
						patchMembers->data.fl[3*count+1] = _Y->data.fl[j*_X->cols + i];
						patchMembers->data.fl[3*count+2] = _Z->data.fl[j*_X->cols + i];
						if (patchMembers->data.fl[3*count+2]<minDepth)
							minDepthInd = count ;
					}
					count++;
				}		

			
			// Optional
			/**************************** Mean Calculation ******************************************/
			/*	float meanX=0, meanY=0, meanZ=0;
				for(int ii=0; ii<pointCount; ii++)
				{
					meanX += patchMembers->data.fl[3*ii  ] ; 
					meanY += patchMembers->data.fl[3*ii+1] ;
					meanZ += patchMembers->data.fl[3*ii+2] ;
				}

				meanX = meanX/ (float)pointCount ;
				meanY = meanY/ (float)pointCount ;
				meanZ = meanZ/ (float)pointCount ;*/

			/************************** Standard Deviation Calculation ******************************/
/*			float standardDevX=0, standardDevY=0, standardDevZ=0;
			for(int ii=0; ii<pointCount; ii++)
			{
				standardDevX += (patchMembers->data.fl[3*ii  ] - meanX)*(patchMembers->data.fl[3*ii  ] - meanX); 
				standardDevY += (patchMembers->data.fl[3*ii+1] - meanY)*(patchMembers->data.fl[3*ii+1] - meanY);
				standardDevZ += (patchMembers->data.fl[3*ii+2] - meanZ)*(patchMembers->data.fl[3*ii+2] - meanZ);
			}
			standardDevX =cvSqrt(standardDevX / (float)pointCount); if(standardDevX == 0){standardDevX = 1;}
			standardDevY =cvSqrt(standardDevY / (float)pointCount); if(standardDevY == 0){standardDevY = 1;}
			standardDevZ =cvSqrt(standardDevZ / (float)pointCount); if(standardDevZ == 0){standardDevZ = 1;} */	
			/**************************** Mean Normalization ******************************************/
			/*for(int ii=0; ii<pointCount; ii++)
			{
				patchMembers->data.fl[3*ii  ] = patchMembers->data.fl[3*ii  ] - meanX; 
				patchMembers->data.fl[3*ii+1] = patchMembers->data.fl[3*ii+1] - meanY;
				patchMembers->data.fl[3*ii+2] = patchMembers->data.fl[3*ii+2] - meanZ;
			}	*/	
			///**************************** Variance Normalization ******************************************/
			/*for(int ii=0; ii<pointCount; ii++)
			{
				patchMembers->data.fl[3*ii  ] = patchMembers->data.fl[3*ii  ] / standardDevX; 
				patchMembers->data.fl[3*ii+1] = patchMembers->data.fl[3*ii+1] / standardDevY;
				patchMembers->data.fl[3*ii+2] = patchMembers->data.fl[3*ii+2] / standardDevZ;
			}*/
			//*******************************************************************************************************************************/
			

			//***************************** Center Shifting ********************************************/			
			CvPoint3D32f point = {patchMembers->data.fl[0], patchMembers->data.fl[1], patchMembers->data.fl[2]};
			//CvPoint3D32f point = {patchMembers->data.fl[count], patchMembers->data.fl[count], patchMembers->data.fl[count]};
			//CvPoint3D32f point = {0, 0, 0}; //use original point coordinates
			//CvPoint3D32f point = {0, 0, minDepth}; //use original point coordinates
			//CvPoint3D32f point = {_X->data.fl[indY*imageSize.width + indX], _Y->data.fl[indY*imageSize.width + indX], _Z->data.fl[indY*imageSize.width + indX]};
			
			/*CvMat* maskX = cvCreateMat( pointCount, 1, CV_32FC1 );
			CvMat* maskY = cvCreateMat( pointCount, 1, CV_32FC1 );
			CvMat* maskZ = cvCreateMat( pointCount, 1, CV_32FC1 );
			double minX,minY,minZ, maxX,maxY,maxZ;
				for(int n=0;n<pointCount;n++)
				{	maskX->data.fl[n ]= patchMembers->data.fl[3*n ] ;
					maskY->data.fl[n ]= patchMembers->data.fl[3*n +1] ;
					maskZ->data.fl[n ]= patchMembers->data.fl[3*n +2] ; }
			cvMinMaxLoc(maskX,&minX,&maxX); cvMinMaxLoc(maskY,&minY,&maxY); cvMinMaxLoc(maskZ,&minZ,&maxZ);
			CvPoint3D32f point = { minX, minY, minZ};*/
			
			for(int ii=0; ii<pointCount; ii++)
			{
				patchMembers->data.fl[3*ii  ] = 1.0*(patchMembers->data.fl[3*ii  ] - point.x); 
				patchMembers->data.fl[3*ii+1] = 1.0*(patchMembers->data.fl[3*ii+1] - point.y);
				patchMembers->data.fl[3*ii+2] = 1.0*(patchMembers->data.fl[3*ii+2] - point.z);
				WLS->data.fl[ii] = patchMembers->data.fl[3*ii+2]; 
			}		
			//******************************************************************************************/

			
			/***************************************Eigenvalue analysis*************************************/
			for(int k=0; k<pointCount; k++)
			{
				D->data.fl[6*k+0] = patchMembers->data.fl[3*k  ]*patchMembers->data.fl[3*k  ];
				D->data.fl[6*k+1] = patchMembers->data.fl[3*k+1]*patchMembers->data.fl[3*k+1];
				D->data.fl[6*k+2] = patchMembers->data.fl[3*k+1]*patchMembers->data.fl[3*k];
				D->data.fl[6*k+3] = patchMembers->data.fl[3*k  ];
				D->data.fl[6*k+4] = patchMembers->data.fl[3*k+1];
				D->data.fl[6*k+5] = 1;
				
			}

			cvSetZero(A);
			cvTranspose( D, DT );
			cvMatMulAdd(DT, D, 0, A);

			CvMat* Ainv = cvCreateMat( 6, 6, CV_32FC1 );
			cvInvert(A,Ainv,CV_SVD);

			CvMat* AF = cvCreateMat( 6, pointCount, CV_32FC1 );
			cvMatMulAdd(Ainv, DT, 0, AF);
			CvMat* F = cvCreateMat( 6, 1, CV_32FC1 );
			cvMatMulAdd(AF, WLS, 0, F);

					
			
			int index = indY*_X->cols + indX;

		
			for(int param = 0; param < 6; param++)
					quadPatchParam->data.fl[6*index + param] = F->data.fl[param];

			cvReleaseMat(&AF);
			cvReleaseMat(&F);
			cvReleaseMat(&Ainv);
		}
	}
	

	cvReleaseMat(&X);
	cvReleaseMat(&Y);

	cvReleaseMat(&patchMembers);
	cvReleaseMat(&A);
	cvReleaseMat(&W);
	cvReleaseMat(&U);
	cvReleaseMat(&V);
	cvReleaseMat(&D);
	cvReleaseMat(&DT);
	cvReleaseMat(&T);
	cvReleaseMat(&WLS);
	cvReleaseMat(&estimate);
	}

void PolyCurvatures(const CvMat* coefficients,const float x, const float y, const float z, float* k1, float* k2, float *sIndex)
// given the quadratic surface coefficients ax^2+by^2+cxy+dx+ey+f=z with "coeffcients" matrix, and a point on the surface p=(x,y,z)
// finds the principal curvatures k1 and k2
{
	float a,b,c,d,e,f;
	a=coefficients->data.fl[0]; b=coefficients->data.fl[1];c=coefficients->data.fl[2];d=coefficients->data.fl[3];
	e=coefficients->data.fl[4];

	float hx,hxx,hy,hyy,hxy;
	hx=2*a*x+c*y+d;
	hxx=2*a;
	hy=2*b*y+c*x+e;
	hyy=2*b;
	hxy=c;

	
	float H,K;

	H=0.5*((1+hx*hx)*hyy-2*hx*hy*hxy+(1+hy*hy)*hxx)/cvSqrt((1+hx*hx+hy*hy)*(1+hx*hx+hy*hy)*(1+hx*hx+hy*hy));
	K=(hxx*hyy-hxy*hxy)/((1+hx*hx+hy*hy)*(1+hx*hx+hy*hy));


	*k1=(H+cvSqrt(H*H-K)); 
	*k2=(H-cvSqrt(H*H-K));
	float temp;
	if (*k1<*k2)
	{
		temp=*k1;
		*k1=*k2;
		*k2=temp;
	}
	
	*sIndex=0.5-(1/CV_PI)*atan2(*k1+*k2,*k1-*k2);
}
