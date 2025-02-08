#include <iostream>
#include <map>
#include <random>
#include <chrono>

int main() {
    // 创建一个map来存储浮点数
    std::map<float, int> floatMap;

    // 随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);

    // 开始计时
    auto start = std::chrono::high_resolution_clock::now();

    // 生成1000个随机浮点数并插入到map中
    for (int i = 0; i < 10000000; ++i) {
        float randomFloat = dis(gen);
        floatMap[randomFloat] = i;  // 使用i作为value，只是为了填充map
    }

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();

    // 计算耗时
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "insert into map cost: " << elapsed.count() << " s" << std::endl;

    return 0;
}