// Basic inclusions.
#include <cstdlib>
#include <cstdio>
#include <iostream>
using namespace std;

#include "solver.h"

int main(int argc, char* argv[])
{
	int w = atoi(argv[1]);
	int h = atoi(argv[2]);
	int iter_num = atoi(argv[3]); // number of iterations
	int function_number = atoi(argv[4]);
	char *filename_results = "", *filename_stat = "";
 	if (argc >= 6)
 		filename_results = argv[5]; // filename to write results
 	if (argc >= 7)
	 	filename_stat = argv[6]; // filename to write stat


	MPI_Init(&argc, &argv);
	
	int nproc, rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);

	Solver solver(w, h, function_number, iter_num, rank, nproc);
	solver.calc_grid_values(0, 0); // init grid values

	MPI_Barrier(MPI_COMM_WORLD);
	double start_time = MPI_Wtime();
	
	solver.solve_numerically();
	
	MPI_Barrier(MPI_COMM_WORLD);
	double end_time = MPI_Wtime();
	double total_time = end_time - start_time;

	solver.send_to_master();
	 if (rank == 0)
	{
	  	solver.write_result_to_file(filename_results);
	  	solver.write_stat_to_file(total_time, -1, filename_stat);	
	}

	MPI_Finalize();

	return 0;
}