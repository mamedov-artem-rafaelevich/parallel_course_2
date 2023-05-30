#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char** argv)
{
	int rank, size;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	int ack;
	if(rank==0)
	{
		printf("I am process %d", rank);
		ack = 1;
		MPI_Send(&ack,1,MPI_INT,0,0,MPI_COMM_WORLD);
	}
	else if(rank==1)
	{
		MPI_Status status;
		MPI_Recv(&ack,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);
		printf("I am process %d\n",rank,ack);
	}
	MPI_Finalize();
	return 0;
}
