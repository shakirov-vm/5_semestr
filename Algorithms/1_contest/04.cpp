#include <iostream>
#include <vector>

class permutations { // want log(n) find and erase

	std::vector<int> vec;
public:
	permutations(int size) {
		vec.reserve(size);
		for(int i = 1; i <= size; i++) vec.push_back(i);
	}
	int get_i_th_bigger(int i_th) {
		
		int pos = 0, arr_runner = 0;

		while(true) {
			if (vec[arr_runner] == 0) {
				arr_runner++;
				continue;
			}
			if (pos == i_th) break;
			pos++;
			arr_runner++;
		}

		int ret = vec[arr_runner];
		vec[arr_runner] = 0;
		return ret;
	}
};

int fact(int x) {
	if (x == 0) return 1;
	int factorial = 1;
	for(; x > 0; x--) {
		factorial *= x;
	}
	return factorial;
}

void orthogonalize(std::vector<int>& vec, int num) {
	
	for(int i = vec.size() - 1; i >= 0; i--) {

		vec[vec.size() - i - 1] = num / fact(i); // push to vec[start - i]
		num -= vec[vec.size() - i - 1] * fact(i); // num %= i!
	}
}

void create_num(std::vector<int>& vec) {

	permutations basket(vec.size());

	for(int i = 0, pos; i < vec.size(); i++) {
		pos = vec[i];
		std::cout << basket.get_i_th_bigger(pos) << " ";
	}
	std::cout << std::endl;
}

int main() {

	int quantity, pos;
	std::cin >> quantity >> pos;
	pos--;
	std::vector<int> vec(quantity);
	orthogonalize(vec, pos);
	create_num(vec);
}