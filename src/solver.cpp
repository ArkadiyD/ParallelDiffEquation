#include "solver.h"

//get index based on row
int Solver::get_index_row(int x, int y)
{
	return x * GLOBAL_GRID_W + y;
}

//get index based on column
int Solver::get_index_col(int x, int y) 
{
	return y * GLOBAL_GRID_H + x;
}

//calculate right part of equation
double Solver::f_value(int iteration, int x, int y)
{
	if (FUNCTION_NUMBER == 1)
		return 0.0;
	else if (FUNCTION_NUMBER == 2)
	{
		double exp = pow(M_E, (iteration * D_T));
		double x_v = x * D_X, y_v = y * D_Y;
		return (x_v * x_v  + y_v * y_v) * sin(x_v * y_v) - pow(M_E, x_v)+ 15;
	}
	return 0.0;
}

//calculate precise solution of equation
double Solver::u_value(double iteration, int x, int y)
{
	double value = 0.0;

	if (FUNCTION_NUMBER == 1)
	{
		double exp = pow(M_E, -iteration * D_T);
		value = exp * (sin(x * D_X) + sin(y * D_Y));
		
	}
	else if (FUNCTION_NUMBER == 2)
	{
		double exp = pow(M_E, iteration * D_T);
		value = exp * sin(x * D_X) * sin(y * D_Y);
		value = pow(M_E, x * D_X) + 15 * iteration * D_T + sin(x * D_X * y * D_Y);
	}

	return value;
}

//util function to print grid values
void Solver::print_grid_values()
{
	for (int x = 0; x < GRID_H; ++x)
	{
		for (int y = 0; y < GRID_W; ++y)
			cout << grid_values[get_index_row(x, y)];
		cout<<endl;
	}
}

void sleep_(unsigned int mseconds)
{
    clock_t goal = mseconds + clock();
    while (goal > clock());
}

//constructor
Solver::Solver(int w, int h, int function_number, int n_iterations, int rank, int nproc)
{
	RANK = rank;
	NPROC = nproc;
	GLOBAL_GRID_W = w;
	GLOBAL_GRID_H = h;
	OMP_THREADS = omp_get_max_threads();

	FUNCTION_NUMBER = function_number;
	N_ITERATIONS = n_iterations;
	if (FUNCTION_NUMBER == 1) // f = 0, u = e^(-t)sin(x) + e^(-t)sin(y)  
	{
		X_MAX = M_PI;
		X_MIN = 0;
		Y_MAX = M_PI;
		Y_MIN = 0;	
	}
	else if (FUNCTION_NUMBER == 2) // f = (x^2 + y^2) * sin(x * y) - e ^ x + 15 , u = e^x + 15 * t+ sin(x * y);
	{
		X_MAX = 1;
		X_MIN = -1;
		Y_MAX = 1;
		Y_MIN = -1;	
	}

	D_X = (X_MAX - X_MIN) / (double)(h - 1);
	D_Y = (Y_MAX - Y_MIN) / (double)(w - 1);
	D_T = 0.0001 / (w * w);

	cout << D_X << ' ' << D_Y << ' ' << D_T << endl;

	if (rank < w % nproc)
	{
		GRID_W = w / nproc + 1;
		proc_y_indices.first = GRID_W * rank;
		proc_y_indices.second = proc_y_indices.first + GRID_W;
	}
	else
	{
		GRID_W = w / nproc;
		proc_y_indices.first = (GRID_W + 1) * (w % nproc) + (rank - w % nproc) * GRID_W;
		proc_y_indices.second = proc_y_indices.first + GRID_W;
	}

	if (rank < h % nproc)
	{
		GRID_H = h / nproc + 1;
		proc_x_indices.first = GRID_H * rank;
		proc_x_indices.second = proc_x_indices.first + GRID_H;
	}
	else
	{
		GRID_H = h / nproc;
		proc_x_indices.first = (GRID_H + 1) * (h % nproc) + (rank - h % nproc) * GRID_H;
		proc_x_indices.second = proc_x_indices.first + GRID_H;
	}

	for (int r = 0; r < NPROC; ++r)
	{
		if (r < h % nproc)
		{
			int G_H = h / nproc + 1;
			dims_x.push_back(G_H * (r + 1));
		}
		else
		{
			int G_H = h / nproc;
			dims_x.push_back( (GRID_H + 1) * (h % nproc) + (r - h % nproc) * G_H + G_H);
		}	

		if (r < w % nproc)
		{
			int G_W = w / nproc + 1;
			dims_y.push_back(G_W * (r + 1));
		}
		else
		{
			int G_W = w / nproc;
			dims_y.push_back((G_W + 1) * (w % nproc) + (r - w % nproc) * G_W + G_W);
		}	
	}

	grid_values = new double[GLOBAL_GRID_H * GLOBAL_GRID_W];
	buf_grid_values = new double[GLOBAL_GRID_H * GLOBAL_GRID_W];

	upper_border = new double[GLOBAL_GRID_W];
	lower_border = new double[GLOBAL_GRID_W];
	left_border = new double[GLOBAL_GRID_H];
	right_border = new double[GLOBAL_GRID_H];

	a_matrix_y = new double[GLOBAL_GRID_W];
	b_matrix_y = new double[GLOBAL_GRID_W];
	c_matrix_y = new double[GLOBAL_GRID_W];
	
	a_matrix_x = new double[GLOBAL_GRID_H];
	b_matrix_x = new double[GLOBAL_GRID_H];
	c_matrix_x = new double[GLOBAL_GRID_H];
	
	A_x = 0.5 * D_T / (D_X * D_X);
	B_x = 1 + D_T / (D_X * D_X);
	C_x = 0.5 * D_T / (D_X * D_X);

	A_y = 0.5 * D_T / (D_Y * D_Y);
	B_y = 1 + D_T / (D_Y * D_Y);
	C_y = 0.5 * D_T / (D_Y * D_Y);

	F_y = new double[GLOBAL_GRID_W];
	P_y = new double[GLOBAL_GRID_W];
	Q_y = new double[GLOBAL_GRID_W];
	


	send_buf_x = new double[GLOBAL_GRID_H * OMP_THREADS];
	recv_buf_x = new double[GLOBAL_GRID_H * OMP_THREADS];
	send_buf_y = new double[GLOBAL_GRID_W * OMP_THREADS];
	recv_buf_y = new double[GLOBAL_GRID_W * OMP_THREADS];

	for (int x = proc_x_indices.first; x < proc_x_indices.second; ++x)
	{
		a_matrix_x[x] = A_x;
		c_matrix_x[x] = C_x;
		b_matrix_x[x] = -B_x;
	}

	for (int y = 0; y < GLOBAL_GRID_W; ++y)
	{
		a_matrix_y[y] = A_y;
		c_matrix_y[y] = C_y;
		b_matrix_y[y] = -B_y;
	}
}

Solver::~Solver()
{
	delete[] grid_values;
	delete[] buf_grid_values;

	delete[] lower_border;
	delete[] upper_border;
	delete[] left_border;
	delete[] right_border;

	delete[] a_matrix_x;
	delete[] b_matrix_x;
	delete[] c_matrix_x;
	delete[] a_matrix_y;
	delete[] b_matrix_y;
	delete[] c_matrix_y;

	delete[] F_y;

	delete[] P_y;
	delete[] Q_y;
}

//calculate precise values in a grid
void Solver::calc_grid_values(int iteration, int mode)
{	
	if (mode == 0)
	{
		for (int y = 0; y < GLOBAL_GRID_W; ++y) 
		{
			for (int x = 0; x < GLOBAL_GRID_H; ++x) 
			{
				grid_values[get_index_row(x, y)] = u_value(iteration, x, y);
				buf_grid_values[get_index_row(x, y)] = u_value(iteration, x, y);
			}
		}
	}
	else if (mode == 1)
	{
		for (int y = 0; y < GLOBAL_GRID_W; ++y) 
		{
			for (int x = 0; x < GLOBAL_GRID_H; ++x) 
				grid_precise_values[get_index_row(x, y)] = u_value(iteration, x, y);	
		}
	}
}

//set precise values in borders
void Solver::set_border_values(int iteration)
{
	for (int y = 0; y < GRID_W; ++y) 
	{
		lower_border[y] = u_value(iteration + 1, 0, y);
		upper_border[y] = u_value(iteration + 1, GRID_H - 1, y);
	}

	for (int x = 0; x < GRID_H; ++x) 
	{
		left_border[x] = u_value(iteration + 1, x, 0);
		right_border[x] = u_value(iteration + 1, x, GRID_W - 1);
	}
}


//perform a single iteration
void Solver::perform_iteration(int iteration)
{
	double buf_value;
	int x,y;
	int prev_proc = RANK - 1, next_proc = RANK + 1;
	int start_y_ind = proc_y_indices.first, start_x_ind = proc_x_indices.first;
	int last_y_ind = proc_y_indices.second, last_x_ind = proc_x_indices.second;
	double send_matrix_buf[OMP_THREADS * 2];
	double recv_matrix_buf[OMP_THREADS * 2];

	double F_x [GLOBAL_GRID_H * OMP_THREADS];
	double P_x [GLOBAL_GRID_H * OMP_THREADS];
	double Q_x [GLOBAL_GRID_H * OMP_THREADS];

	if (RANK == 0)
		prev_proc = MPI_PROC_NULL;
	if (RANK == NPROC - 1)
		next_proc = MPI_PROC_NULL;
	
	MPI_Status status;


	//send last to next, recv prev-1 from prev
	if (RANK < NPROC - 1)
	{
		#pragma omp parallel for shared(send_buf_y)
		for (y = 0; y < GLOBAL_GRID_W; ++y)
			send_buf_y[y] = grid_values[get_index_row(proc_x_indices.second, y)];
		
		MPI_Send(send_buf_y, GLOBAL_GRID_W, MPI_DOUBLE, next_proc, 0, MPI_COMM_WORLD);
	}

	if (RANK > 0)
	{
		MPI_Recv(recv_buf_y, GLOBAL_GRID_W, MPI_DOUBLE, prev_proc, 0, MPI_COMM_WORLD, &status);

		for (y = 0; y < GLOBAL_GRID_W; ++y)
			grid_values[get_index_row(proc_x_indices.first - 1, y)] = recv_buf_y[y];
	}

	//send first to prev, recv last+1 from next
	if (RANK > 0)
	{
		#pragma omp parallel for shared(send_buf_y)
		for (y = 0; y < GLOBAL_GRID_W; ++y)
			send_buf_y[y] = grid_values[get_index_row(proc_x_indices.first, y)];
		
		MPI_Send(send_buf_y, GLOBAL_GRID_W, MPI_DOUBLE, prev_proc, 0, MPI_COMM_WORLD);
	}

	if (RANK < NPROC - 1)
	{
		MPI_Recv(recv_buf_y, GLOBAL_GRID_W, MPI_DOUBLE, next_proc, 0, MPI_COMM_WORLD, &status);

		for (y = 0; y < GLOBAL_GRID_W; ++y)
			grid_values[get_index_row(proc_x_indices.second, y)] = recv_buf_y[y];
	}

	////////////////////////////////////////////////////////////////////////////
	int r = 0;
	while (r < GLOBAL_GRID_W)
	{
		if (r == 0)
		{
			for (x = proc_x_indices.first; x < proc_x_indices.second; ++x)
				buf_grid_values[get_index_row(x, 0)] = u_value(iteration + 0.5, x, 0);		
		}

		if (r + OMP_THREADS >= GLOBAL_GRID_W - 2)
		{
			for (x = proc_x_indices.first; x < proc_x_indices.second; ++x)
				buf_grid_values[get_index_row(x, GLOBAL_GRID_W - 1)] = u_value(iteration + 0.5, x, GLOBAL_GRID_W - 1);		
		}

		#pragma omp parallel private(y)
		for (y = max(1, r); y < min(GLOBAL_GRID_W - 1, r + OMP_THREADS); ++y)
		{
			for (x = proc_x_indices.first; x < proc_x_indices.second; ++x)
			{
				double L = (grid_values[get_index_row(x, y - 1)]  - 2 * grid_values[get_index_row(x, y)] + grid_values[get_index_row(x, y + 1)] ) / (D_Y * D_Y);   
				F_x[get_index_col(x, y-r)] = -grid_values[get_index_row(x, y)] - 0.5 * D_T * (f_value(iteration, x, y) + L);
			}
		}

		if (RANK > 0)
			MPI_Recv(recv_matrix_buf, 2 * OMP_THREADS, MPI_DOUBLE, prev_proc, 0, MPI_COMM_WORLD, &status);
		
		for (y = max(1, r); y < min(GLOBAL_GRID_W - 1, r + OMP_THREADS); ++y)
		{
			if (RANK > 0)
			{
				P_x[get_index_col(proc_x_indices.first - 1, y-r)] = recv_matrix_buf[(y - r) * 2 + 0];
				Q_x[get_index_col(proc_x_indices.first - 1, y-r)] = recv_matrix_buf[(y - r) * 2 + 1];
			}
			else
			{
				P_x[get_index_col(proc_x_indices.first, y-r)] = -c_matrix_x[0] / b_matrix_x[0];
				Q_x[get_index_col(proc_x_indices.first, y-r)] = F_x[get_index_col(0, y-r)] / b_matrix_x[0];
			}	
		}

		int start_x = proc_x_indices.first;
		if (start_x == 0) start_x = 1;
		int last_x = proc_x_indices.second;
		for (y = max(1, r); y < min(GLOBAL_GRID_W - 1, r + OMP_THREADS); ++y)
		{
			for (x = start_x; x < last_x; ++x)
			{
				P_x[get_index_col(x, y-r)] = c_matrix_x[x] / ( -a_matrix_x[x] * P_x[get_index_col(x-1, y-r)] - b_matrix_x[x]);
				Q_x[get_index_col(x, y-r)] = (-F_x[get_index_col(x, y-r)] + a_matrix_x[x] * Q_x[get_index_col(x-1, y-r)]) / (-a_matrix_x[x] * P_x[get_index_col(x-1, y-r)] - b_matrix_x[x]);
			}
			
			send_matrix_buf[(y - r) * 2 + 0] = P_x[get_index_col(proc_x_indices.second - 1, y-r)];
			send_matrix_buf[(y - r) * 2 + 1] = Q_x[get_index_col(proc_x_indices.second - 1, y-r)];
		}

		MPI_Send(send_matrix_buf, 2 * OMP_THREADS, MPI_DOUBLE, next_proc, 0, MPI_COMM_WORLD);

		//send to prev
		if (RANK > 0)
		{
			for (y = max(1, r); y < min(GLOBAL_GRID_W - 1, r + OMP_THREADS); ++y)
				send_matrix_buf[y-r] = buf_grid_values[get_index_row(proc_x_indices.first, y)];
			
			MPI_Send(&send_matrix_buf, OMP_THREADS, MPI_DOUBLE, prev_proc, 0, MPI_COMM_WORLD);
		}

		if (RANK < NPROC - 1)
		{ 
			MPI_Recv(recv_matrix_buf, OMP_THREADS, MPI_DOUBLE, next_proc, 0, MPI_COMM_WORLD, &status);
			
			for (y = max(1, r); y < min(GLOBAL_GRID_W - 1, r + OMP_THREADS); ++y)
				buf_grid_values[get_index_row(proc_x_indices.second, y)] = recv_matrix_buf[y-r];
		}

		for (y = max(1, r); y < min(GLOBAL_GRID_W - 1, r + OMP_THREADS); ++y)
		{
			for (x = proc_x_indices.second - 1; x >= proc_x_indices.first; --x)
			{
				if (x == 0 || x == GLOBAL_GRID_H - 1)
				{
					buf_grid_values[get_index_row(x, y)] = u_value(iteration + 0.5, x, y);
					continue;
				}
			
				buf_grid_values[get_index_row(x, y)] = P_x[get_index_col(x, y-r)] * buf_grid_values[get_index_row(x + 1, y)] + Q_x[get_index_col(x, y-r)];
			}
		}

		r += OMP_THREADS;
	}
	
	//-----------------------------------------------------------------------
	// cout <<  proc_x_indices.first << ' ' << proc_x_indices.second << endl;
	// for (y = 0; y < GLOBAL_GRID_W; ++y)
	// {
	// 	for (x = proc_x_indices.first; x < proc_x_indices.second; ++x)
	// 		grid_values[get_index_row(x,y)] = buf_grid_values[get_index_row(x,y)];
	// }
	//-----------------------------------------------------------------------

	//send last to next, recv prev-1 from prev
	if (RANK < NPROC - 1)
	{
		for (y = 0; y < GLOBAL_GRID_W; ++y)
			send_buf_y[y] = buf_grid_values[get_index_row(proc_x_indices.second - 1, y)];
		
		MPI_Send(send_buf_y, GLOBAL_GRID_W, MPI_DOUBLE, next_proc, 0, MPI_COMM_WORLD);
	}

	if (RANK > 0)
	{
		MPI_Recv(recv_buf_y, GLOBAL_GRID_W, MPI_DOUBLE, prev_proc, 0, MPI_COMM_WORLD, &status);

		for (y = 0; y < GLOBAL_GRID_W; ++y)
			buf_grid_values[get_index_row(proc_x_indices.first - 1, y)] = recv_buf_y[y];
	}

	//send first to prev, recv last+1 from next
	if (RANK > 0)
	{
    	for (y = 0; y < GLOBAL_GRID_W; ++y)
			send_buf_y[y] = buf_grid_values[get_index_row(proc_x_indices.first, y)];
		
		MPI_Send(send_buf_y, GLOBAL_GRID_W, MPI_DOUBLE, prev_proc, 0, MPI_COMM_WORLD);
	}


	if (RANK < NPROC - 1)
	{
		MPI_Recv(recv_buf_y, GLOBAL_GRID_W, MPI_DOUBLE, next_proc, 0, MPI_COMM_WORLD, &status);

		for (y = 0; y < GLOBAL_GRID_W; ++y)
			buf_grid_values[get_index_row(proc_x_indices.second, y)] = recv_buf_y[y];
	}
	
	////////////////////////////////////////////////////////////////////////////

	for (x = proc_x_indices.first; x < proc_x_indices.second; ++x)
	{
		if (x == 0 || x == GLOBAL_GRID_H - 1)
		{
			for (y = 0; y < GLOBAL_GRID_W; ++y)
				grid_values[get_index_row(x, y)] = u_value(iteration + 1, x, y);
			
			continue;
		}
		
		
		for (y = 0; y < GLOBAL_GRID_W; ++y)
		{
			double L = (buf_grid_values[get_index_row(x - 1, y)]  - 2 * buf_grid_values[get_index_row(x, y)] + buf_grid_values[get_index_row(x + 1, y)] ) / (D_X * D_X);   
			F_y[y] = -buf_grid_values[get_index_row(x, y)] - 0.5 * D_T * (f_value(iteration, x, y) + L);
		}

		P_y[0] = -c_matrix_y[0] / b_matrix_y[0];
		Q_y[0] = F_y[0] / b_matrix_y[0];
		
		for (y = 1; y < GLOBAL_GRID_W; ++y)
		{
			P_y[y] = c_matrix_y[y] / ( -a_matrix_y[y] * P_y[y - 1] - b_matrix_y[y]);
			Q_y[y] = (-F_y[y] + a_matrix_y[y] * Q_y[y - 1]) / (-a_matrix_y[y] * P_y[y - 1] - b_matrix_y[y]);
		}

		for (y = GLOBAL_GRID_W - 1; y >= 0; --y)
		{
			if (y == 0 || y == GLOBAL_GRID_W - 1)
			{
				grid_values[get_index_row(x, y)] = u_value(iteration + 1, x, y);
				continue;
			}
			grid_values[get_index_row(x, y)] = P_y[y] * grid_values[get_index_row(x, y + 1)] + Q_y[y];
		}
	}
}

//calculate numerical solution in a loop
void Solver::solve_numerically()
{
	for (int iter = 0; iter < N_ITERATIONS; ++iter)
	{
		perform_iteration(iter);
	}
}

//gather all values on a master node
void Solver::send_to_master()
{
	MPI_Status status;

	if (RANK > 0)
	{
		int sizes[2] = {proc_x_indices.first, proc_x_indices.second};
		MPI_Send(&sizes, 2, MPI_INT, 0, 0, MPI_COMM_WORLD);
		MPI_Send(&grid_values[get_index_row(proc_x_indices.first, 0)], (proc_x_indices.second - proc_x_indices.first) * GLOBAL_GRID_W, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}
	else
	{
		int sizes[NPROC][2];
		for (int r = 1; r < NPROC; ++r)
		{
			MPI_Recv(&sizes[r], 2, MPI_INT, r, 0, MPI_COMM_WORLD, &status);
			MPI_Recv(&grid_values[get_index_row(sizes[r][0], 0)], (sizes[r][1] - sizes[r][0]) * GLOBAL_GRID_W, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, &status);
		}
	}
}

//write running statistics to a file
void Solver::write_stat_to_file(double total_time, double error, char *filename)
{
	if (filename == "")
	{
		cout << "GRID SIZE: " << GLOBAL_GRID_W << " " << GLOBAL_GRID_H << endl;
		cout << "ITERATIONS: " << N_ITERATIONS << endl;
		cout << "TIME: " << total_time << endl;
		cout << "ERROR: " << error << endl;
	}
	else
	{
		fstream out_file;
		out_file.open(filename, std::ios::out);
		
		out_file << "GRID SIZE: " << GLOBAL_GRID_W << " " << GLOBAL_GRID_H << endl;
		out_file << "ITERATIONS: " << N_ITERATIONS << endl;
		out_file << "OMP THREADS: " << OMP_THREADS << endl;
		out_file << "TIME: " << total_time << endl;
		out_file << "ERROR: " << error << endl;

		out_file.close();
	}
}

//write solution to a file
void Solver::write_result_to_file(char *filename)
{
	if (filename == "")
	{
		for (int x = 0; x < GLOBAL_GRID_H; ++x) 
		{
			for (int y = 0; y < GLOBAL_GRID_W; ++y) 
				if (abs(grid_values[get_index_row(x, y)] - grid_precise_values[get_index_row(x, y)]) > 0.000001)
					cout << x << ' ' << y << ' ' << get_index_row(x, y) << ' ' << grid_values[get_index_row(x, y)] << "~" << grid_precise_values[get_index_row(x, y)]  << endl;
		}
	}
	else
	{
		fstream out_file;
		out_file.open(filename, ios::out);
		for (int x = 0; x < GLOBAL_GRID_H; ++x) 
		{
			for (int y = 0; y < GLOBAL_GRID_W; ++y) 
				out_file << grid_values[get_index_row(x, y)] << " ";
			out_file << endl;
		}
		out_file.close();
	}
}

//calculate error of numerical solution
double Solver::calc_error()
{
	double mean_sqrt_error = 0.0;
	for (int x = 0; x < GLOBAL_GRID_H; ++x) 
	{
		for (int y = 0; y < GLOBAL_GRID_W; ++y) 
		{
			double diff = (grid_values[get_index_row(x, y)] - grid_precise_values[get_index_row(x, y)]); 
			mean_sqrt_error += diff * diff;
		}
	}
	mean_sqrt_error /= (GLOBAL_GRID_H * GLOBAL_GRID_W);
	mean_sqrt_error = sqrt(mean_sqrt_error);
	return mean_sqrt_error;
}