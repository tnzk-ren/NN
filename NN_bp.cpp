#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>

#define MAX 1000
#define N 10

double sigmoid(double x)
{
	return 1.0 / (1 + exp(-x));
}

double sig_d(double x)
{
	return sigmoid(x) * (1 - sigmoid(x));
}

double loss(double y, double t)
{
	return (y - t) * (y - t) / 2.0;
}

class MLP {
public:
	MLP(int _m, double _al, double _eps);
	double *forward(double x[]);
	void backward(double t[]);

private:
	double alpha;
	double eps;
	double W1[MAX][MAX];
	double W2[MAX][MAX];
	double dW1[MAX][MAX];
	double dW2[MAX][MAX];
	int i_unit;
	int m_unit;
	int o_unit;
	double l1[MAX];
	double l2[MAX];
	double l3[MAX];
	double o1[MAX];
	double o2[MAX];
	double o3[MAX];
};

MLP::MLP(int _m, double _al, double _eps)
{
	i_unit = 2;
	m_unit = _m;
	o_unit = 1;
	alpha = _al;
	eps = _eps;

	std::random_device seed;
	std::mt19937 engine(seed());         
										
	double mu = 0.0;
	double sigma = 1.0;
	std::normal_distribution<> dist(mu, sigma);

	double rate = 0.1;
	for (int i = 0; i < i_unit; i++) {
		for (int j = 0; j < m_unit; j++) {
			W1[i][j] = rate * dist(engine);
			dW1[i][j] = 0;
		}
	}
	for (int i = 0; i < m_unit; i++) {
		for (int j = 0; j < o_unit; j++) {
			W2[i][j] = rate * dist(engine);
			dW2[i][j] = 0;
		}
	}

}

double *MLP::forward(double x[])
{
	for (int i = 0; i < i_unit; i++) {
		l1[i] = x[i];
		o1[i] = 1 * x[i];
	}
	
	for (int i = 0; i < m_unit; i++) {
		l2[i] = 0;
		for (int j = 0; j < i_unit; j++) {
			l2[i] += W1[j][i] * o1[j];
		}
		o2[i] = sigmoid(l2[i]);
	}

	for (int i = 0; i < o_unit; i++) {
		l3[i] = 0;
		for (int j = 0; j < m_unit; j++) {
			l3[i] += W2[j][i] * o2[j];
		}
		o3[i] = 1 * l3[i];
	}

	return o3;
}

void MLP::backward(double t[])
{
	double d3[MAX];
	d3[0] = (o3[0] - t[0]);
	for (int i = 0; i < m_unit; i++) {
		for (int j = 0; j < o_unit; j++) {
			dW2[i][j] = alpha * dW2[i][j] - eps * d3[j] * o2[i];
		}
	}
	
	double d2[MAX];
	for (int i = 0; i < m_unit; i++) {
		d2[i] = 0;
		for (int j = 0; j < o_unit; j++) {
			d2[i] += W2[i][j] * d3[j] * sig_d(l2[i]);
		}
	}
	for (int i = 0; i < i_unit; i++) {
		for (int j = 0; j < m_unit; j++) {
			dW1[i][j] = alpha * dW1[i][j] - eps * d2[j] * o1[i];
		}
	}

	for (int i = 0; i < i_unit; i++) {
		for (int j = 0; j < m_unit; j++) {
			W1[i][j] += dW1[i][j];
		}
	}
	for (int i = 0; i < m_unit; i++) {
		for (int j = 0; j < o_unit; j++) {
			W2[i][j] += dW2[i][j];
		}
	}
}
int main(void)
{
	double x[N][2] = {0};
	double t[N][1] = { 0 };
	double alpha = 0.7;
	double eps = 0.01;
	MLP *model = new MLP(10, alpha, eps);

	std::vector<int> vec(N);
	for (int i = 0; i < N; i++) {
		x[i][0] = i * 2.0 *3.141592 / N;
		x[i][1] = 1.0;
		t[i][0] = 10 * sin(x[i][0]);

		vec[i] = i;
	}

	for (int k = 0; k < 1000; k++) {
		std::mt19937 get_rand_mt;
		std::shuffle(vec.begin(), vec.end(), get_rand_mt);

		for (int j = 0; j < N; j++) {
			int i = vec[j];
			double y;
			y = model->forward(x[i])[0];
			model->backward(t[i]);
		}
	}

	for (int i = 0; i < N * 10; i++) {
		double y;
		double tx[2];
		tx[0] = i * 2 * 3.141592 / (N * 10);
		tx[1] = 1.0;
		double ty = 10 * sin(tx[0]);
		y = model->forward(tx)[0];
		std::cout << tx[0] << "," << ty << "," << y << std::endl;
	}
	return 0;
}