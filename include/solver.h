#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>
#include <omp.h>
#include <cstring>
#include <algorithm>
#include <vector>
using namespace std;


class Solver
{
private:
	int RANK, NPROC;
	int OMP_THREADS;
	int FUNCTION_NUMBER, N_ITERATIONS;
	double X_MIN, X_MAX, Y_MIN, Y_MAX;
	double D_T, D_X, D_Y;
	int GRID_W, GRID_H, GLOBAL_GRID_W, GLOBAL_GRID_H; 
	pair<int, int> proc_x_indices;
	pair<int, int> proc_y_indices;
	vector<int> dims_x;
	vector<int> dims_y;

	double *grid_precise_values;
	double *grid_values;
	double *buf_grid_values;

	double  *lower_border, *upper_border;
	double *left_border, *right_border;

	double *a_matrix_x, *b_matrix_x, *c_matrix_x;
	double *a_matrix_y, *b_matrix_y, *c_matrix_y;

	double A_x,	B_x, C_x;
	double A_y,	B_y, C_y;

	double *F_y;
	double  *P_y, *Q_y;
	double *recv_buf_x, *send_buf_x, *send_buf_y, *recv_buf_y;

	double f_value(int iteration_number, int x, int y); //calculate right part of equation
	double u_value(double iteration_number, int x, int y); //calculate precise solution of equation
	void print_grid_values(); //util function to print grid values
	int get_index_row(int x, int y); //get index based on row
	int get_index_col(int x, int y); //get index based on column
	
public:
	Solver(int w, int h, int function_number, int n_iterations, int rank, int nproc);
	~Solver();

	void calc_grid_values(int iteration, int mode); //calculate precise values in a grid
	void set_border_values(int iteration); //set precise values in borders
	void perform_iteration(int iteration); //perform a single iteration
	void solve_numerically(); //calculate numerical solution in a loop
	void send_to_master(); //gather all values on a master node

	void write_stat_to_file(double total_time, double error, char *filename = ""); //write running statistics to a file
	void write_result_to_file(char *filename = ""); //write solution to a file
	double calc_error(); //calculate error of numerical solution
};