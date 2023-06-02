#include <stdio.h>
#include <stdlib.h>

int main()
{
        FILE* fp = fopen("start.bin","wb+");
        double array[1024];
        for(int i=0; i<1024; i++)
	{
                array[i]=((double)(rand()%10))/100;
		printf("%f ",array[i]);
	}
	printf("\n");
        fwrite(array,sizeof(double),1024,fp);
        fclose(fp);
        // fp = fopen("fc1.bin","wb+");
        // double fc1[1024*256];
        // for(int i=0; i<1024*256; i++)
	// {
        //         fc1[i]=((double)(rand()%10))/100;
	// 	printf("%f ",fc1[i]);fp = fopen("fc1.bin","wb+");
        // double fc1[1024*256];
        // for(int i=0; i<1024*256; i++)
	// {
        //         fc1[i]=((double)(rand()%10))/100;
	// 	printf("%f ",fc1[i]);
	// }
	// printf("\n");
        // fwrite(fc1,sizeof(double),1024*256,fp);
        // fclose(fp);
        // fp = fopen("fc2.bin","wb+");
        // double fc2[256*16];
        // for(int i=0; i<16*256; i++)
	// {
        //         fc2[i]=((double)(rand()%10))/100;
	// 	printf("%f ",fc2[i]);
	// }
	// printf("\n");
        // fwrite(fc2,sizeof(double),16*256,fp);
        // fclose(fp);
        // fp = fopen("fc3.bin","wb+");
        // double fc3[16];
        // for(int i=0; i<16; i++)
	// {
        //         fc3[i]=((double)(rand()%10))/100;
	// 	printf("%f ",fc3[i]);
	// }
	// printf("\n");
        // fwrite(fc3,sizeof(double),16,fp);
        // fclose(fp);
	// }
	// printf("\n");
        // fwrite(fc1,sizeof(double),1024*256,fp);
        // fclose(fp);
        // fp = fopen("fc2.bin","wb+");
        // double fc2[256*16];
        // for(int i=0; i<16*256; i++)
	// {
        //         fc2[i]=((double)(rand()%10))/100;
	// 	printf("%f ",fc2[i]);
	// }
	// printf("\n");
        // fwrite(fc2,sizeof(double),16*256,fp);
        // fclose(fp);
        // fp = fopen("fc3.bin","wb+");
        // double fc3[16];
        // for(int i=0; i<16; i++)
	// {
        //         fc3[i]=((double)(rand()%10))/100;
	// 	printf("%f ",fc3[i]);
	// }
	// printf("\n");
        // fwrite(fc3,sizeof(double),16,fp);
        // fclose(fp);
        return 0;
}
