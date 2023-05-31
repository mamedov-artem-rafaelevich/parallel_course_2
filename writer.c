#include <stdio.h>
#include <stdlib.h>

int main()
{
        FILE* fp = fopen("start.bin","wb+");
        float array[1024];
        for(int i=0; i<1024; i++)
                array[i]=(float)(rand()%255)/255;
        fwrite(&array,sizeof(float),1024,fp);
        fclose(fp);
        fp = fopen("fc1.bin","wb+");
        float fc1[1024*256];
        for(int i=0; i<1024*256; i++)
                fc1[i]=(float)(rand()%255)/255;
        fwrite(&fc1,sizeof(float),1024*256,fp);
        fclose(fp);
        fp = fopen("fc2.bin","wb+");
        float fc2[256*16];
        for(int i=0; i<16*256; i++)
                fc2[i]=(float)(rand()%255)/255;
        fwrite(&fc2,sizeof(float),16*256,fp);
        fclose(fp);
        fp = fopen("fc3.bin","wb+");
        float fc3[16];
        for(int i=0; i<16; i++)
                fc3[i]=(float)(rand()%255)/255;
        fwrite(&fc3,sizeof(float),16,fp);
        fclose(fp);
        return 0;
}
