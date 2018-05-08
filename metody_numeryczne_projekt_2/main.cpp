// Autor: Jonasz Rudnik 165480

#include <ctime>
#include <iostream>

int index[6] = { 1,6,5,4,8,0 };

#define index(n) index[n-1]
#define WRITEOUTINFUNCTIONS true


clock_t start;

class Matrix
{
	const int width;
	const int height;
	double **data;

	double** alloc_array(const int &height, const int &width) const
	{
		double **result = new double*[height];
		for (int i = 0; i < height; i++)
		{
			result[i] = new double[width];
		}
		return result;
	}

public:
	double** get_data()
	{
		return data;
	}
	
	int get_width()
	{
		return width;
	}
	int get_height()
	{
		return height;
	}

	void copy_contents_from(Matrix &source)
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				data[y][x] = source[y][x];
			}
		}
	}

	void copy_upper_from(Matrix &source)
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = y + 1; x < width; x++)
			{
				data[y][x] = source[y][x];
			}
		}
	}

	void copy_lower_from(Matrix &source)
	{
		for (int y = 1; y < height; y++)
		{
			for (int x = 0; x < y; x++)
			{
				data[y][x] = source[y][x];
			}
		}
	}

	void copy_diagonal_from(Matrix &source)
	{
		if (height != source.height || width != source.width) throw std::invalid_argument("matrix sizes don't match");
		if (height != width) throw std::invalid_argument("width != height");
		
		for (int i = 0; i < height; i++)
		{		
			data[i][i] = source[i][i];			
		}
	}
	
	double* operator [] (int i) const
	{
		return data[i];
	}
	
	Matrix* operator + (Matrix* matrix)
	{		
		return this->add(matrix);
	}
		

	Matrix *add(Matrix *matrix)
	{
		Matrix *result = new Matrix(height, width);
		result->copy_contents_from(*this);
		
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				(*result)[y][x] = data[y][x] + (*matrix)[y][x];
			}
		}		
		return result;
	}


	void multiply_it(double multiplier)
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				data[y][x] *= multiplier;
			}
		}
	}

	Matrix *subtract(Matrix *matrix)
	{
		Matrix *temp = matrix->multiply(-1);
		Matrix *result = this->add(temp);
		return result;
	}

	Matrix *new_matrix_with_same_size()
	{
		return new Matrix(height, width);
	}

	Matrix *add(double n)
	{
		Matrix *result = new Matrix(height, width);
		result->copy_contents_from(*this);

		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				(*result)[y][x] = data[y][x] + n;
			}
		}
		return result;
	}

	Matrix *subtract(double n)
	{
		return this->add(-n);
	}

	Matrix *multiply(Matrix &matrix)
	{
		if (this->width != matrix.height) 
		{			
			throw std::invalid_argument("Matrix_1 width different than Matrix_2 height");			
		}
		Matrix *result = new Matrix(height, matrix.width);
		
		for (int row = 0; row < height; row++)
		{
			for (int column = 0; column < matrix.width; column++)
			{
				double sum = 0;
				for (int i = 0; i < height; i++)
				{
					sum += data[row][i] * matrix[i][column];
				}
				(*result)[row][column] = sum;
			}
		}
		return result;
	}

	Matrix *multiply(double multiplier)
	{
		Matrix* result = new_matrix_with_same_size();
		result->copy_contents_from(*this);
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				(*result)[y][x] *= multiplier;
			}
		}
		return result;
	}

	void multiply_row(int row, double multiplier)
	{
		for (int x = 0; x < width; x++)
		{
			data[row][x] *= multiplier;
		}
	}

	void row_add(int source, int target, double multiplier = 1)
	{
		for (int x = 0; x < width; x++)
		{
			data[target][x] += data[source][x] * multiplier;
		}
	}
	

	Matrix* odwrotnosc()
	{
		Matrix *left = new_matrix_with_same_size();
		left->copy_contents_from(*this);
		
		Matrix *right = new_matrix_with_same_size();
		right->fill(0);
		right->fill_diagonal(1);
		
		for (int y = 0; y < height; y++)
		{
			double multiplier1 = 1 / (*left)[y][y];
			left->multiply_row(y, multiplier1);
			right->multiply_row(y, multiplier1);
			
			for (int y2 = 0; y2 < height; y2++)
			{
				if (y != y2)
				{
					double multiplier2 = -(*left)[y2][y];
					left->row_add(y, y2, multiplier2);
					right->row_add(y, y2, multiplier2);
				}
			}
		}		
		delete left;
		return right;
	}

	Matrix* odwrotnosc_na_diagonali()
	{
		Matrix *result = new_matrix_with_same_size();		
		result->fill(0);
		for (int xy = 0; xy < height; xy++)
		{
			result->get_data()[xy][xy] = 1 / get_data()[xy][xy];
		}		
		return result;
	}

	Matrix(const int &height, const int &width) : width(width), height(height)
	{
		data = alloc_array(height, width);
	}

	~Matrix()
	{
		for (int i = 0; i < height; i++)
		{
			delete[] data[i];
		}
		delete[] data;
	}

	void out() const
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				printf("%*f ", 2, data[y][x]);
			}
			printf("\n");
		}
	}

	void fill(const double &value = 0) const
	{
		for (int y = 0; y < height; y++)
		{
			std::fill(data[y], data[y] + width, value);
		}
	}

	void fill_diagonal(const double &value, const int shift = 0) const
	{
		int x = 0, y = 0;
		if (shift >= 0) x += shift;
		else y -= shift;

		for (; x < width && y < height; x++, y++)
		{
			data[y][x] = value;
		}
	}	
};

Matrix *wektor_residuum(Matrix &A, Matrix &x, Matrix &b)
{
	Matrix *iloczyn = A.multiply(x);
	Matrix *result = iloczyn->subtract(&b);
	delete iloczyn;
	return result;
}

double norma_residuum(Matrix &wektor_residuum)
{
	double suma = 0;
	for (int i = 0; i < wektor_residuum.get_height(); i++)
	{
		suma += wektor_residuum[i][0] * wektor_residuum[i][0];
	}
	return sqrt(suma);
}

Matrix* jacobi(Matrix &A, Matrix &b)
{
	Matrix *L = A.new_matrix_with_same_size();
	L->fill(0);
	L->copy_lower_from(A);	
	L->multiply_it(-1);

	Matrix *U = A.new_matrix_with_same_size();
	U->fill(0);
	U->copy_upper_from(A);	
	U->multiply_it(-1);

	Matrix *D = A.new_matrix_with_same_size();
	D->fill(0);
	D->copy_diagonal_from(A);		


	Matrix *Dm1 = D->odwrotnosc_na_diagonali();		
	Matrix *LpU = L->add(U);
	Matrix *x = new Matrix(A.get_height(), 1);
	x->fill(1);

	Matrix *Dm1_t_b = Dm1->multiply(b);

	//Here we go with iterations
	for (int i = 0; i < 100; i++)
	{		
		Matrix *LpUx = LpU->multiply(*x);
		Matrix *left_side = Dm1->multiply(*LpUx);
		Matrix *x2 = left_side->add(Dm1_t_b);
		delete LpUx;
		delete left_side;
		delete x;
		x = x2;

		Matrix* residuum_v = wektor_residuum(A, *x, b);
		double n_residuum = norma_residuum(*residuum_v);
		delete residuum_v;
		if (n_residuum <= 10e-9)
		{
			if(WRITEOUTINFUNCTIONS)
				std::cout << "Do metody jakobiego bylo wymagane " << i << " iteracji.\n";
			break;
		}
		if (i == 99)
		{
			if (WRITEOUTINFUNCTIONS)
				std::cout << "Metoda jakobiego nie zbiegla sie w ciagu 100 iteracji" << std::endl;
		}
	}	

	delete Dm1_t_b;
	delete L;
	delete U;
	delete D;
	delete LpU;
	delete Dm1;

	return x;
}



Matrix* gauss_seidel(Matrix &A, Matrix &b)
{
	Matrix *L = A.new_matrix_with_same_size();
	L->fill(0);
	L->copy_lower_from(A);
	L->multiply_it(-1);

	Matrix *U = A.new_matrix_with_same_size();
	U->fill(0);
	U->copy_upper_from(A);
	U->multiply_it(-1);

	Matrix *D = A.new_matrix_with_same_size();
	D->fill(0);
	D->copy_diagonal_from(A);

	Matrix *Dm1 = D->odwrotnosc_na_diagonali();
	Matrix *LpU = L->add(U);

	Matrix *x = new Matrix(A.get_height(), 1);
	x->fill(1);

	Matrix *Dm1_t_b = Dm1->multiply(b);

	//Here we go with iterations
	for (int i = 0; i < 100; i++)
	{
		for (int row = 0; row < LpU->get_height(); row++)
		{
			double weighted_sum = 0;
			for (int shift = 0; shift < LpU->get_width(); shift++)
			{
				weighted_sum += LpU->get_data()[row][shift] * x->get_data()[shift][0];
			}
			x->get_data()[row][0] = Dm1->get_data()[row][row] * weighted_sum + Dm1_t_b->get_data()[row][0];
		}

		Matrix* w_residuum = wektor_residuum(A, *x, b);
		double n_residuum = norma_residuum(*w_residuum);
		delete w_residuum;
	
		if (n_residuum <= 10e-9)
		{
			if (WRITEOUTINFUNCTIONS)
				std::cout << "Do metody gaussa-seidla bylo wymagane " << i << " iteracji.\n";
			break;
		}
		if (i == 99)
		{
			if (WRITEOUTINFUNCTIONS)
				std::cout << "Metoda gaussa-seidla nie zbiegla sie w ciagu 100 iteracji" << std::endl;
		}
	}

	delete L;
	delete U;
	delete D;
	delete Dm1_t_b;
	delete LpU;
	delete Dm1;

	return x;
}

Matrix* faktoryzacjaLU(Matrix* A, Matrix *b)
{
	Matrix* U = A->new_matrix_with_same_size();
	U->copy_contents_from(*A);

	Matrix* L = A->new_matrix_with_same_size();
	L->fill(0);
	L->fill_diagonal(1);

	
	for (int k = 0; k < A->get_width() - 1; k++)
	{
		for (int j = k+1; j < A->get_width(); j++)
		{
			L->get_data()[j][k] = U->get_data()[j][k] / U->get_data()[k][k];
			for (int k2 = k; k2 < A->get_width(); k2++)
			{
				U->get_data()[j][k2] = U->get_data()[j][k2] - L->get_data()[j][k] * U->get_data()[k][k2];
			}
		}
	}

	Matrix *y = b->new_matrix_with_same_size();
	//Ly=b     podstawianie wprzód
	for (int i = 0; i < y->get_height(); i++)
	{		
		double odjemnik = 0;
		for (int j = 0; j < i; j++)
		{
			odjemnik += y->get_data()[j][0] * L->get_data()[i][j];
		}
		y->get_data()[i][0] = (b->get_data()[i][0] - odjemnik) / L->get_data()[i][i];
	}
	Matrix *x = y->new_matrix_with_same_size();	
	
	//Ux=y     podstawianie wstecz
	for (int i = x->get_height() - 1; i >= 0; i--)
	{
		double odjemnik = 0;
		for (int j = x->get_height() - 1; j > i; j--)
		{
			odjemnik += x->get_data()[j][0] * U->get_data()[i][j];
		}
		x->get_data()[i][0] = (y->get_data()[i][0] - odjemnik) / U->get_data()[i][i];
	}

	return x;
}


int main()
{
	const bool 
	do_B = 1,//true, 
	do_C = 0,//true, 
	do_D = 0,//true,
	do_E = 0;
	
	
	start = clock();

	//Zadanie A
	std::cout << "A" << std::endl;
	double a1, a2, a3;
	a1 = 5 + index(4);
	a2 = a3 = -1;
	const int N = 900 + index(5) * 10 + index(6);

		
	Matrix A(N, N);
	A.fill(0);	
	A.fill_diagonal(a1, 0);
	A.fill_diagonal(a2, 1);
	A.fill_diagonal(a2, -1);
	A.fill_diagonal(a3, 2);
	A.fill_diagonal(a3, -2);
	
	Matrix b(N, 1);
	for (int n = 0; n < N; n++)
	{
		b[n][0] = sin((n + 1)*(index(3) + 1));
	}

	//Zadanie B
	if(do_B)
	{		
		std::cout << "B" << std::endl;
		clock_t start = clock();
		Matrix* x = jacobi(A, b);
		std::cout << "Jacobi zakonczyl sie po " << (clock() - start) / 1000.0 << "s dzialania.\n";
		delete x;

		start = clock();
		x = gauss_seidel(A, b);
		std::cout << "Gauss-Seidel zakonczyl sie po " << (clock() - start) / 1000.0 << "s dzialania.\n";		
	}
	
	//Zadanie C
	A.fill_diagonal(3);	
	if (do_C) {	
		std::cout << "C" << std::endl;
		jacobi(A, b);
		gauss_seidel(A, b);		
	}//Można zauważyć że obie meteody nie zbiegają się dla tego przykładu

	//Zadanie D
	if(do_D)
	{
		std::cout << "D" << std::endl;
		start = clock();
		Matrix* x = faktoryzacjaLU(&A, &b);
		std::cout << "metoda faktoryzacji LU zakonczyla sie po " << (clock() - start) / 1000.0 << "s dzialania.\n";
		Matrix *residuum_vector = wektor_residuum(A, *x, b);
		std::cout << "Norma residuum x wyliczonego metoda faktoryzacji LU to: " << norma_residuum(*residuum_vector) << std::endl;
		delete residuum_vector;
	}

	//Zadanie E
	if(do_E)
	{
		const bool dojacobi = 0;
		const bool dogauss_seidel = 1;
		const bool dolu = 0;

		std::cout << "E" << std::endl;
		const int count = 8;
		int Ns[count] = { 100,500,1000,2000,3000,4000,5000,6000};
	//
	//	(100, 0.028)(500, 0.623)(1000, 2.441)(2000, 9.943)(3000, 22.882)(4000, 39.665)(5000, 62.647)
		
		//(100, 0.028)(500, 0.623)(1000, 2.441)(2000, 9.943)
		for (int N : Ns)
		{
			//std::cout << "Jacobi dla macierzy " << N << "x" << N << std::endl;
			Matrix A(N, N);
			A.fill(0);
			A.fill_diagonal(a1, 0);
			A.fill_diagonal(a2, 1);
			A.fill_diagonal(a2, -1);
			A.fill_diagonal(a3, 2);
			A.fill_diagonal(a3, -2);

			Matrix b(N, 1);
			for (int n = 0; n < N; n++)
			{
				b[n][0] = sin((n + 1)*(index(3) + 1));
			}
			Matrix* x;
			clock_t start = clock();
			if (dojacobi)
			{
				x = jacobi(A, b);
			}
			if (dogauss_seidel)
			{
				x = gauss_seidel(A, b);
			}
			if(dolu)
			{
				x = faktoryzacjaLU(&A, &b);
			}
			//std::cout << "Jacobi dla macierzy " << N << "x" << N << " zakonczyl sie po " << (clock() - start) / 1000.0 << "s dzialania.\n";
			std::cout << "(" << N << "," << (clock() - start) / 1000.0 << ")";
			delete x;
			
		}
	}

	std::cout << "Program ukonczyl dzialanie po " << (clock() - start) / 1000.0 << "s.\n";
	system("pause");
	return 0;
}