#include <iostream>
#include <vector>
#include <cmath>
float ndegree(float d, int i)
{
	for(int k=0; k<i; k++)
		d/=10;
	return d;
};

float readFloat(char* str)
{
	float ans=0;
	char f=1, f1=0;
	int j=0;
	for(int i=0; str[i]; i++)
	{
		if(str[0]=='-')
		{
			if(i==0)
				i++;
			f++;
		}
		if(str[i]=='.')
			f1++;
		if(f1==1)
			j++;
		if(str[i]>=48 && str[i]<58 && f1==0)
			ans=ans*10+float(str[i]);
		else if(str[i]>=48 && str[i]<58 && f1==1)
			ans+=ndegree(float(str[i]),j);
	}
	return ans;
};

class float3
{
	public:
	float x,y,z;
#pragma acc routine seq
	void set(const float3* f)
	{
		this->x=f->x;
		this->y=f->y;
		this->z=f->z;
	};
};

int main(int argc, char** argv)
{
	float a=0;
	float s=0;
	float n=0;
	if(argv[1][1]=='h')
	{
		std::cout << "Put -h to show this.\n";
		std::cout << "Put -a <NUMBER_OF_ACCURACY*10^6> -s <SIZE^2> -n <NUMBER_OF_ITERATION*10^6>.\n";
	}
	else
	{
		for(int k=1; k<argc; k+=2)
		{
			if(argv[k][1]=='a')
				a=readFloat(argv[k+1]);
			else if(argv[k][1]=='s')
				s=readFloat(argv[k+1]);
			else if(argv[k][1]=='n')
				n=readFloat(argv[k+1]);
		}
		std::vector<std::vector float> setka;
		setka.resize(s);
#pragma acc kernels
		for(int i=0; i<s; i++)
			setka[i].resize(s);
		setka[0][0]=10;
		setka[0][s-1]=20;
		setka[s-1][0]=30;
		setka[s-1][s-1]=20;
		float p=10/s;
		p=(float(int(1000000*p)%int(1000000*a)))/1000000;
		float p1=p*2;
		p1=(float(int(1000000*p1)%int(1000000*a)))/1000000;
#pragma acc kernels
		for(int j=1; j<s; j++)
			setka[0][j]=setka[0][j-1]+p;
#pragma acc kernels
		for(int j=1; j<s; j++)
			setka[j][0]=setka[j-1][0]+p1;
#pragma acc kernels
		for(int j=1; j<s; j++)
			setka[s-1][j]=setka[s-1][j]+p;
#pragma acc kernels
		for(int j=1; j<s; j++)
			setka[j][s-1]=setka[j][s-1]+p;
#pragma acc kernels
		for(int i=1; i<s-1; i++)
			for(int j=1; j<s-1; j++)
				setka[i][j]=std::sqrt(setka[i-1][j]*setka[i-1][j]+setka[i][j-1]*setka[i][j-1]);
		std::cout << (s-1)*(s-1)+5*(s-1) << setka[s-1][s-1]-setka[s-2][s-2] << "\n";
	}
	return 0;
}
