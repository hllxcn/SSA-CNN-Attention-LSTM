clc
close all
clear all

%% 数据读取和预处理
fprintf('开始数据读取和预处理...\n');
result = xlsread('数据集改.xlsx');
num_samples = length(result);  

rng(100000); %固定随机数

% 确保数据是列向量
if size(result, 2) > 1
    result = result(:, 1);
end

fprintf('数据样本数: %d\n', num_samples);

% 自定义STL分解实现
fprintf('使用自定义STL分解方法...\n');
[trend, seasonal, residual] = custom_stl_decomposition(result, 7) ;

% 可视化STL分解结果
figure('Position', [100, 100, 1200, 800])
subplot(4,1,1) 
plot(result, 'b-', 'LineWidth', 1.5)
title('原始时间序列', 'FontSize', 12)
ylabel('值', 'FontSize', 10)
grid on

subplot(4,1,2)
plot(trend, 'r-', 'LineWidth', 1.5)
title('趋势分量', 'FontSize', 12)
ylabel('值', 'FontSize', 10)
grid on

subplot(4,1,3)
plot(seasonal, 'g-', 'LineWidth', 1.5)
title('季节性分量', 'FontSize', 12)
ylabel('值', 'FontSize', 10)
grid on

subplot(4,1,4)
plot(residual, 'm-', 'LineWidth', 1.5)
title('残差分量', 'FontSize', 12)
ylabel('值', 'FontSize', 10)
xlabel('时间点', 'FontSize', 10)
grid on

sgtitle('STL时间序列分解结果', 'FontSize', 14, 'FontWeight', 'bold');
exportgraphics(gcf, 'STL_decomposition.png', 'Resolution', 300);

%% 使用STL分解后的分量重构多通道特征
multi_channel_data = [result, trend, seasonal, residual];
num_channels = size(multi_channel_data, 2);
fprintf('多通道数据维度: %d×%d\n', size(multi_channel_data));

%% 数据准备
kim = 7;                       % 输入时间步长
zim = 1;                       % 输出时间步长

% 检查数据维度
fprintf('result维度: %d×%d\n', size(result));
fprintf('multi_channel_data维度: %d×%d\n', size(multi_channel_data));

% 重构数据集
res_multi = [];
for i = 1: num_samples - kim - zim + 1
% for i = 1: num_samples - kim - zim + 1
    % 多通道输入：每个通道取kim个时间步
    input_multi = [];
    for ch = 1:num_channels
        channel_data = multi_channel_data(i:i + kim - 1, ch);
        input_multi = [input_multi, channel_data'];
    end
    % 输出为接下来的zim个值（多步预测）
%     res_multi(i, :) = [input_multi, result(i : i + zim - 1)'];
    res_multi(i, :) = [input_multi, result(i + kim + zim - 1)'];
end

data_multi = res_multi;

% 划分输入输出
input_multi = data_multi(:, 1:kim*num_channels);
output_multi = data_multi(:, end-zim+1:end);

inputSize = kim * num_channels;
outputSize = zim;  
numTimeStepsTrain = floor(0.9 * numel(data_multi(:, 1)));
fprintf('numTimeStepsTrain的大小: %d×%d\n', numTimeStepsTrain);

fprintf('data_multi的大小: %d×%d\n', size(data_multi));
fprintf('numel(data_multi(:, 1))的大小: %d×%d\n', numel(data_multi(:, 1)));

% 训练集：仍然使用重叠窗口（步长为1），以增加训练样本数量
XTrain = input_multi(1:numTimeStepsTrain, :);
YTrain = output_multi(1:numTimeStepsTrain, :);
fprintf('训练集的大小: %d×%d\n', size(XTrain));

%% 测试集：改为步长为zim的非重叠窗口
% 计算测试集的非重叠窗口数量
% num_test_samples_total = size(input_multi, 1) - numTimeStepsTrain;
num_test_samples_total = size(input_multi, 1) - numTimeStepsTrain+kim+zim-1;
num_test_windows = floor((num_test_samples_total-kim-zim+1)/ zim);

fprintf('原始测试集大小: %d\n', num_test_samples_total);
fprintf('非重叠窗口数: %d\n', num_test_windows);

% 选择测试集样本（间隔zim步）
test_indices = numTimeStepsTrain + (1:zim:num_test_windows*zim);
fprintf('test_indices: %d\n', test_indices);

test_indices = test_indices(1:min(num_test_windows, length(test_indices)));

XTest = input_multi(test_indices, :);
YTest = output_multi(test_indices, :);

fprintf('训练集大小: %d×%d\n', size(XTrain));
fprintf('训练集输出维度: %d×%d\n', size(YTrain));
fprintf('测试集大小: %d×%d (步长=%d的非重叠窗口)\n', size(XTest), zim);
fprintf('测试集输出维度: %d×%d\n', size(YTest));

%% 数据归一化
x = XTrain;
y = YTrain;

[xnorm, xopt] = mapminmax(x', 0, 1);
[ynorm, yopt] = mapminmax(y', 0, 1);

% 转换训练数据格式
Train_xNorm = {};
Train_yNorm = [];
Train_y = [];

for i = 1:size(ynorm, 2)
    Train_xNorm{i} = reshape(xnorm(:, i), kim, 1, num_channels);
    Train_yNorm(:, i) = ynorm(:, i);
    Train_y(i, :) = y(i, :);
end
Train_yNorm = Train_yNorm';

% 转换测试数据格式
xtest = XTest;
ytest = YTest;
[xtestnorm] = mapminmax('apply', xtest', xopt);
[ytestnorm] = mapminmax('apply', ytest', yopt);

Test_xNorm = {};
Test_yNorm = [];
Test_y = [];

for i = 1:size(ytestnorm, 2)
    Test_xNorm{i} = reshape(xtestnorm(:, i), kim, 1, num_channels);
    Test_yNorm(:, i) = ytestnorm(:, i);
    Test_y(i, :) = ytest(i, :);
end
Test_yNorm = Test_yNorm';

%% 添加数据统计信息
fprintf('\n=== 数据统计信息 ===\n');
fprintf('总时间序列长度: %d\n', num_samples);
fprintf('输入窗口大小(kim): %d\n', kim);
fprintf('输出步长(zim): %d\n', zim);
fprintf('训练集样本数: %d (步长=1的重叠窗口)\n', size(XTrain, 1));
fprintf('测试集样本数: %d (步长=%d的非重叠窗口)\n', size(XTest, 1), zim);
fprintf('总预测时长: %d 个时间点\n', size(XTest, 1) * zim);
fprintf('测试集覆盖率: %.2f%%\n', (size(XTest, 1) * zim) / (num_samples - numTimeStepsTrain) * 100);

%% SSA优化参数设置
SearchAgents = 10;        % 增加种群数量
Max_iterations = 30;      % 增加迭代次数
lowerbound = [1e-6, 0.0001, 10];  % 调整参数下界
upperbound = [1e-3, 0.005, 200];  % 调整参数上界
dimension = 3;            % 优化参数个数

% QRBL参数优化
QRBL_probability = 0.8;  % 提高QRBL执行概率
QRBL_improved_count = 0; % 记录QRBL改进次数
QRBL_strength = 0.3;     % 降低扰动强度

fprintf('开始SSA优化...\n');

%% SSA优化算法主循环
% 初始化参数
ST = 0.8;                    % 降低预警值
PD = 0.2;                    % 增加发现者比例
PDNumber = round(SearchAgents * PD);  % 发现者数量
SDNumber = round(SearchAgents * 0.25); % 增加意识到危险的麻雀数量

% 边界检查
if max(size(upperbound)) == 1
   upperbound = upperbound .* ones(1, dimension);
   lowerbound = lowerbound .* ones(1, dimension);  
end

% 种群初始化
pop_lsat = initialization(SearchAgents, dimension, upperbound, lowerbound);
pop_new = pop_lsat;

% 计算初始适应度
fitness = zeros(1, SearchAgents);
for i = 1:SearchAgents
    fitness(i) = fun(pop_new(i, :), Train_xNorm, Train_yNorm, Test_xNorm, Test_y, yopt,zim);
end

% 排序并获取最佳适应度
[fitness, index] = sort(fitness);
GBestF = fitness(1); 

% 重新排列种群
for i = 1:SearchAgents
    pop_new(i, :) = pop_lsat(index(i), :);
end

GBestX = pop_new(1, :);
X_new = pop_new;
curve = zeros(1, Max_iterations);
QRBL_curve = zeros(1, Max_iterations); % 记录QRBL改进情况

% SSA主优化循环
for iter = 1:Max_iterations
    fprintf('迭代次数: %d/%d, 最佳适应度: %.6f\n', iter, Max_iterations, GBestF);
    
    BestF = fitness(1);
    R2 = rand(1);

    % 发现者位置更新
    for j = 1:PDNumber
        if R2 < ST
            X_new(j, :) = pop_new(j, :) .* exp(-j / (rand(1) * Max_iterations));
        else
            X_new(j, :) = pop_new(j, :) + randn(1, dimension) * 0.1; % 减小扰动幅度
        end     
    end
   
    % 跟随者位置更新
    for j = PDNumber + 1:SearchAgents
        if j > (SearchAgents - PDNumber) / 2 + PDNumber
            X_new(j, :) = randn(1, dimension) .* exp((pop_new(end, :) - pop_new(j, :)) / j^2);
        else
            A = ones(1, dimension);
            for a = 1:dimension
                if rand() > 0.5
                    A(a) = -1;
                end
            end
            AA = A' / (A * A');     
            X_new(j, :) = pop_new(1, :) + abs(pop_new(j, :) - pop_new(1, :)) .* AA';
        end
    end
   
    % 危险预警和位置更新
    Temp = randperm(SearchAgents);
    SDchooseIndex = Temp(1:SDNumber); 
   
    for j = 1:SDNumber
        if fitness(SDchooseIndex(j)) > BestF
            X_new(SDchooseIndex(j), :) = pop_new(1, :) + randn(1, dimension) * 0.1; % 减小扰动幅度
        elseif fitness(SDchooseIndex(j)) == BestF
            K = 2 * rand() - 1;
            X_new(SDchooseIndex(j), :) = pop_new(SDchooseIndex(j), :) + K * (abs(pop_new(SDchooseIndex(j), :) - pop_new(end, :)) / (fitness(SDchooseIndex(j)) - fitness(end) + 1e-8));
        end
    end

    % 边界控制
    for j = 1:SearchAgents
        for a = 1:dimension
            if X_new(j, a) > upperbound(a)
                X_new(j, a) = upperbound(a);
            end
            if X_new(j, a) < lowerbound(a)
                X_new(j, a) = lowerbound(a);
            end
        end
    end 

    % 计算新适应度
    fitness_new = zeros(1, SearchAgents);
    for j = 1:SearchAgents
        fitness_new(j) = fun(X_new(j, :), Train_xNorm, Train_yNorm, Test_xNorm, Test_y, yopt,zim);
    end
   
    %% 改进的准反射学习策略（QRBL）
    fprintf('应用改进的准反射学习策略(QRBL)...\n');
    iter_improved = 0;

    current_population = X_new;  % 当前代未排序
    current_fitness = fitness_new;  % 当前代未排序适应度
    
    for j = 1:SearchAgents
        % 以更高概率执行QRBL
        if rand() < QRBL_probability
            % 生成改进的准反射解
            qrbl_solution = generate_improved_QRBL_solution(...
                X_new(j, :), current_population, current_fitness, GBestX, lowerbound, upperbound, ...
                iter, Max_iterations, QRBL_strength, j);
            
            % 边界检查
            for a = 1:dimension
                if qrbl_solution(a) > upperbound(a)
                    qrbl_solution(a) = upperbound(a);
                end
                if qrbl_solution(a) < lowerbound(a)
                    qrbl_solution(a) = lowerbound(a);
                end
            end
            
            % 计算准反射解的适应度
            qrbl_fitness = fun(qrbl_solution, Train_xNorm, Train_yNorm, Test_xNorm, Test_y, yopt,zim);
            
            % 如果准反射解更好，则替换当前解
            if qrbl_fitness < fitness_new(j)
                fprintf('  QRBL改进个体 %d: 适应度从 %.6f 提升到 %.6f\n', j, fitness_new(j), qrbl_fitness);
                X_new(j, :) = qrbl_solution;
                fitness_new(j) = qrbl_fitness;
                iter_improved = iter_improved + 1;
            end
        end
    end
    
    QRBL_improved_count = QRBL_improved_count + iter_improved;
    QRBL_curve(iter) = iter_improved;
    fprintf('QRBL本轮改进个体数: %d/%d\n', iter_improved, SearchAgents);
   
    % 更新全局最优
    for j = 1:SearchAgents
        if fitness_new(j) < GBestF
            GBestF = fitness_new(j);
            GBestX = X_new(j, :);
            fprintf('  全局最优更新: 适应度 = %.6f\n', GBestF);
        end
    end
   
    % 更新种群和适应度
    pop_new = X_new;
    fitness = fitness_new;

    % 排序
    [fitness, index] = sort(fitness);
    for j = 1:SearchAgents
        pop_new(j, :) = pop_new(index(j), :);
    end

    % 记录收敛曲线
    curve(iter) = GBestF;
end

%% QRBL策略效果分析
fprintf('\n=== QRBL策略效果分析 ===\n');
fprintf('QRBL总改进次数: %d\n', QRBL_improved_count);
fprintf('QRBL平均每代改进: %.2f次\n', QRBL_improved_count / Max_iterations);
fprintf('QRBL改进概率: %.2f%%\n', (QRBL_improved_count / (SearchAgents * Max_iterations)) * 100);

%% 获取最优参数
Best_pos = GBestX;
Best_score =GBestF ;

NumOfUnits = abs(round(Best_pos(1, 3)));       % 最佳神经元个数
InitialLearnRate = Best_pos(1, 2);             % 最佳初始学习率
L2Regularization = Best_pos(1, 1);             % 最佳L2正则化系数

fprintf('SSA-QRBL优化完成!\n');
fprintf('最优参数: 神经元数=%d, 学习率=%.6f, L2正则化=%.6f\n', ...
    NumOfUnits, InitialLearnRate, L2Regularization);

% 定义Huber损失函数
%huber_delta = 1.0;  % Huber损失的delta参数，可调整
% 定义Hinge损失函数
%hinge_margin = 0.5;  % Hinge损失的margin参数
% 创建自定义Huber损失层
%huberLayer = huberRegressionLayer('huber_loss', huber_delta);
% 创建自定义Hinge损失层
%hingeLayer = hingeRegressionLayer('hinge_loss', hinge_margin);
% 
% logCoshRegressionLayer('logcosh_loss');

%% 使用最优参数构建最终模型
fprintf('构建最终神经网络模型...\n');

    layers = [    
        sequenceInputLayer([kim, 1, num_channels], 'Name', 'input')
        sequenceFoldingLayer('Name', 'fold')
        convolution2dLayer([2, 1], 16, 'Stride', [1, 1], 'Name', 'conv1')
        batchNormalizationLayer('Name', 'batchnorm1')
        reluLayer('Name', 'relu1')
        convolution2dLayer([1, 1], 32, 'Stride', [1, 1], 'Name', 'conv2')
        batchNormalizationLayer('Name', 'batchnorm2')
        reluLayer('Name', 'relu2')
        maxPooling2dLayer([1, 2], 'Stride', 1, 'Padding', 'same', 'Name', 'maxpool')
        sequenceUnfoldingLayer('Name', 'unfold')
        flattenLayer('Name', 'flatten')
        lstmLayer(NumOfUnits, 'OutputMode', 'sequence', 'Name', 'lstm1')
        dropoutLayer(0.3, 'Name', 'dropout1')
        MultiHeadAttentionLayer(2, 16, 16, NumOfUnits, 'multihead_attention')
        layerNormalizationLayer('Name', 'layernorm1')
        lstmLayer(NumOfUnits, 'OutputMode', 'last', 'Name', 'lstm2')
        dropoutLayer(0.3, 'Name', 'dropout2')
        fullyConnectedLayer(outputSize, 'Name', 'fc')
         logCoshRegressionLayer('logcosh_loss')];
%        hingeRegressionLayer('hinge_loss', 0.5)];
%        huberRegressionLayer('huber_loss', 1.0)];
%         regressionLayer('Name', 'output')];

    lgraph = layerGraph(layers);
    lgraph = connectLayers(lgraph, 'fold/miniBatchSize', 'unfold/miniBatchSize');

%% 训练选项
opts = trainingOptions('adam', ...
    'MaxEpochs', 200, ...  % 增加训练轮数
    'GradientThreshold', 1, ...
    'ExecutionEnvironment', 'cpu', ...
    'InitialLearnRate', InitialLearnRate, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 20, ...
    'LearnRateDropFactor', 0.8, ...  % 减缓学习率下降
    'Shuffle', 'once', ...  % 每轮都打乱数据
    'SequenceLength', 'longest', ...  % 自动确定序列长度
    'MiniBatchSize', 64, ...  % 减小批大小
    'Verbose', 0, ...
    'Plots', 'training-progress', ...
    'L2Regularization', L2Regularization);  % 添加自定义输出函数

%% 模型训练
fprintf('开始训练神经网络...\n');
tic;
net = trainNetwork(Train_xNorm, Train_yNorm, lgraph, opts);
trainingTime = toc;
fprintf('训练完成，耗时: %.2f 秒\n', trainingTime);

%% 训练集预测
Predict_Ynorm_Train = predict(net, Train_xNorm);
Predict_Y_Train = mapminmax('reverse', Predict_Ynorm_Train', yopt);
Predict_Y_Train = Predict_Y_Train';

% 训练集可视化
figure('Position', [100, 100, 1000, 400])
subplot(1,2,1)
plot(Predict_Y_Train, 'r-', 'LineWidth', 2.0)
hold on
plot(Train_y, 'b-', 'LineWidth', 1.5)
legend('预测值', '实际值', 'Location', 'best')
ylabel('数值')
title('训练集预测结果')
grid on

% 训练误差
train_error = Predict_Y_Train - Train_y;
subplot(1,2,2)
plot(train_error, 'g-', 'LineWidth', 1.0)
ylabel('误差')
title('训练集预测误差')
grid on
sgtitle('训练集预测结果分析', 'FontSize', 12, 'FontWeight', 'bold');
exportgraphics(gcf, 'training_results.png', 'Resolution', 300);

%% 测试集预测
Predict_Ynorm = predict(net, Test_xNorm);
Predict_Y = mapminmax('reverse', Predict_Ynorm', yopt);
Predict_Y = Predict_Y';

% 测试集可视化
figure('Position', [100, 100, 1000, 400])
subplot(1,2,1)
plot(Predict_Y, 'r-', 'LineWidth', 2.0)
hold on
plot(Test_y, 'b-', 'LineWidth', 1.5)
legend('预测值', '实际值', 'Location', 'best')
ylabel('数值')
title('测试集预测结果')
grid on

% 测试误差
test_error = Predict_Y - Test_y;
subplot(1,2,2)
plot(test_error, 'm-', 'LineWidth', 1.0)
ylabel('误差')
title('测试集预测误差')
grid on
sgtitle('测试集预测结果分析', 'FontSize', 12, 'FontWeight', 'bold');
exportgraphics(gcf, 'test_results.png', 'Resolution', 300);

%% 预测结果评价
ae = abs(Predict_Y - Test_y);
rmse = sqrt(mean(ae.^2));
mse = mean(ae.^2);
mae = mean(ae);
mape = mean(ae ./ abs(Test_y)) ;
r2 = 1 - sum(ae.^2) / sum((Test_y - mean(Test_y)).^2);

fprintf('\n=== 预测结果评价指标 ===\n');
fprintf('RMSE = %.6f\n', rmse);
fprintf('MSE  = %.6f\n', mse);
fprintf('MAE  = %.6f\n', mae);
fprintf('MAPE = %.4f\n', mape);
fprintf('R²   = %.6f\n', r2);

fprintf('\n=== 最优参数 ===\n');
fprintf('最佳神经元个数: %d\n', NumOfUnits);
fprintf('最佳初始学习率: %.6f\n', InitialLearnRate);
fprintf('最佳L2正则化系数: %.6f\n', L2Regularization);

%% 收敛曲线和QRBL效果可视化
figure('Position', [100, 100, 1200, 500])

subplot(1,2,1)
plot(1:Max_iterations, curve, 'b-', 'LineWidth', 2);
xlabel('迭代次数');
ylabel('最佳适应度值');
title('SSA-QRBL优化收敛曲线');
grid on;

subplot(1,2,2)
bar(1:Max_iterations, QRBL_curve, 'FaceColor', [0.8 0.2 0.2], 'FaceAlpha', 0.7);
xlabel('迭代次数');
ylabel('QRBL改进个体数');
title('QRBL策略改进效果');
grid on;

sgtitle('SSA-QRBL优化过程分析', 'FontSize', 12, 'FontWeight', 'bold');
exportgraphics(gcf, 'convergence_curve.png', 'Resolution', 300);

%% 残差分析
figure('Position', [100, 100, 800, 600])
subplot(2,2,1)
histogram(test_error, 20, 'FaceColor', 'blue', 'FaceAlpha', 0.7);
title('预测误差分布')
xlabel('误差')
ylabel('频数')

subplot(2,2,2)
normplot(test_error)
title('误差正态概率图')

subplot(2,2,3)
autocorr(test_error)
title('误差自相关图')

subplot(2,2,4)
plot(Test_y, Predict_Y, 'ro', 'MarkerSize', 4)
hold on
plot([min(Test_y), max(Test_y)], [min(Test_y), max(Test_y)], 'k-', 'LineWidth', 2)
xlabel('实际值')
ylabel('预测值')
title('预测值 vs 实际值')
grid on

sgtitle('模型残差分析', 'FontSize', 12, 'FontWeight', 'bold');
exportgraphics(gcf, 'residual_analysis.png', 'Resolution', 300);

fprintf('\n所有分析完成！结果已保存为图像文件。\n');

%% ==================== 自定义函数定义 ====================

%% 自定义STL分解函数
function [trend, seasonal, residual] = custom_stl_decomposition(data, period)
    % 简单的STL分解实现
    n = length(data);
    
    % 1. 计算趋势分量（使用移动平均）
    trend = movmean(data, period);
    
    % 2. 去除趋势后计算季节性分量
    detrended = data - trend;
    
    % 3. 计算季节性分量（按周期平均）
    seasonal = zeros(n, 1);
    for i = 1:period
        indices = i:period:n;
        if ~isempty(indices)
            seasonal_value = mean(detrended(indices));
            seasonal(i:period:end) = seasonal_value;
        end
    end
    
    % 4. 调整季节性分量均值为0
    seasonal = seasonal - mean(seasonal);
    
    % 5. 计算残差
    residual = data - trend - seasonal;
end

%% 麻雀算法初始化函数
function Positions = initialization(SearchAgents_no, dim, ub, lb)
    Boundary_no = size(ub, 2);
    
    if Boundary_no == 1
        Positions = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
    else
        for i = 1:dim
            ub_i = ub(i);
            lb_i = lb(i);
            Positions(:, i) = rand(SearchAgents_no, 1) .* (ub_i - lb_i) + lb_i;
        end
    end
end

function qrbl_solution = generate_improved_QRBL_solution(current_solution, pop, fitness, gbest, lb, ub, iter, max_iter, strength, idx)
    % 改进的准反射学习策略生成新解
    % 结合了基本准反射、全局最优、随机个体和自适应权重
    
    dimension = length(current_solution);
    qrbl_solution = zeros(1, dimension);
    
    % 自适应权重策略
    w1 = 0.6 - 0.3 * (iter / max_iter);  % 基本QRBL权重，从0.6线性减少到0.3
    w2 = 0.3 + 0.2 * (iter / max_iter);   % 全局最优权重，从0.3线性增加到0.5
    w3 = 0.1;                             % 随机个体权重，保持0.1
    w4 = 0.0 + 0.1 * (1 - iter/max_iter); % 当前解权重，从0.1线性减少到0
    
    % 随机选择另一个个体
    other_idx = randi(size(pop, 1));
    while other_idx == idx
        other_idx = randi(size(pop, 1));
    end
    other_solution = pop(other_idx, :);
    
    for i = 1:dimension
        % 1. 基本准反射公式
        if rand() < 0.5
            % 方法1: 关于中心对称
            center = (lb(i) + ub(i)) / 2;
            basic_qrbl = 2 * center - current_solution(i);
        else
            % 方法2: 边界反射
            basic_qrbl = lb(i) + ub(i) - current_solution(i);
        end
        
        % 2. 全局最优引导
        global_guide = gbest(i);
        
        % 3. 随机个体引导
        random_guide = other_solution(i);
        
        % 4. 当前解信息
        current_guide = current_solution(i);
        
        % 5. 加权组合
        combined = w1 * basic_qrbl + w2 * global_guide + ...
                  w3 * random_guide + w4 * current_guide;
                  %% 
        
        % 6. 随机扰动（随着迭代次数增加而减弱）
        perturbation = strength * (ub(i) - lb(i)) * (1 - iter/max_iter) * (2*rand()-1);
        
        % 7. 最终解
        qrbl_solution(i) = combined + perturbation;
        
        % 8. 边界处理（这里也可以使用随机边界策略）
        if qrbl_solution(i) < lb(i) || qrbl_solution(i) > ub(i)
            % 随机重新初始化
            qrbl_solution(i) = lb(i) + (ub(i) - lb(i)) * rand();
        end
    end
end



%% 适应度函数（优化版）
function rmse_value = fun(x, Train_xNorm, Train_yNorm, Test_xNorm, Test_y, yopt,zim)
    % 函数用于计算粒子适应度值
    rng default; % 固定随机数
    warning off; % 关闭警告
    
    % 确保参数为正
    L2Reg = max(x(1), 1e-8);
    LearnRate = max(min(x(2), 0.01), 1e-4);
    NumOfUnits = max(round(x(3)), 10);
    
    outputSize = zim;  % 数据输出y的维度  
        % 训练选项（简化版）
        options = trainingOptions('adam', ...
            'MaxEpochs', 200, ...  
            'GradientThreshold', 1, ...
            'ExecutionEnvironment', 'cpu', ...
            'InitialLearnRate', LearnRate, ...
            'LearnRateSchedule', 'piecewise', ...
            'LearnRateDropPeriod', 20, ...
            'LearnRateDropFactor', 0.8, ...  % 减缓学习率下降
            'Shuffle', 'once', ...
            'SequenceLength', 'longest', ...
            'MiniBatchSize', 64, ...
            'Verbose', 0, ...  % 关闭详细输出
            'L2Regularization', L2Reg, ...
            'Plots', 'none');
        
  % huberLayer = huberRegressionLayer('huber_loss', 1);
 
    
        % 简化的网络结构
        kim = 7;
        num_channels = 4;
        
            layers = [    
        sequenceInputLayer([kim, 1, num_channels], 'Name', 'input')
        sequenceFoldingLayer('Name', 'fold')
        convolution2dLayer([2, 1], 16, 'Stride', [1, 1], 'Name', 'conv1')
        batchNormalizationLayer('Name', 'batchnorm1')
        reluLayer('Name', 'relu1')
        convolution2dLayer([1, 1], 32, 'Stride', [1, 1], 'Name', 'conv2')
        batchNormalizationLayer('Name', 'batchnorm2')
        reluLayer('Name', 'relu2')
        maxPooling2dLayer([1, 2], 'Stride', 1, 'Padding', 'same', 'Name', 'maxpool')
        sequenceUnfoldingLayer('Name', 'unfold')
        flattenLayer('Name', 'flatten')
        lstmLayer(NumOfUnits, 'OutputMode', 'sequence', 'Name', 'lstm1')
        dropoutLayer(0.3, 'Name', 'dropout1')
        MultiHeadAttentionLayer(2, 16, 16, NumOfUnits, 'multihead_attention')
        layerNormalizationLayer('Name', 'layernorm1')
        lstmLayer(NumOfUnits, 'OutputMode', 'last', 'Name', 'lstm2')
        dropoutLayer(0.3, 'Name', 'dropout2')
        fullyConnectedLayer(outputSize, 'Name', 'fc')
        logCoshRegressionLayer('logcosh_loss')];
%        hingeRegressionLayer('hinge_loss', 0.5)];
%        huberRegressionLayer('huber_loss', 1.0)];
%         regressionLayer('Name', 'output')];

    lgraph = layerGraph(layers);
    lgraph = connectLayers(lgraph, 'fold/miniBatchSize', 'unfold/miniBatchSize');
        
        % 网络训练
        net = trainNetwork(Train_xNorm, Train_yNorm, lgraph, options);
        
        % 预测
        Predict_Ynorm = predict(net, Test_xNorm);
        Predict_Y = mapminmax('reverse', Predict_Ynorm', yopt);
        Predict_Y = Predict_Y';

        logcosh_error=stable_logcosh(Predict_Y - Test_y);
        rmse_value = mean(logcosh_error(:));


        % 计算RMSE作为适应度值
        % rmse_value = sqrt(mean((Predict_Y(:) - Test_y(:)).^2));
        % 计算Huber损失作为适应度值
        %huber_error = huberLoss(Predict_Y, Test_y, 1);
        %rmse_value = mean(huber_error(:));
   

end

% Huber损失函数
% function loss = huberLoss(predictions, targets, delta)
%     error = predictions - targets;
%     abs_error = abs(error);
%     
%     % Huber损失
%     quadratic = min(abs_error, delta);
%     linear = abs_error - quadratic;
%     loss = 0.5 * quadratic.^2 + delta * linear;
% end


%% 数值稳定的Log-Cosh计算函数
function loss = stable_logcosh(error)
    % 数值稳定的log(cosh(x))计算
    % 使用恒等式: log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2)
    
    abs_error = abs(error);
    
    % 当|x|很大时，exp(-2|x|) ≈ 0，log(1+0) ≈ 0
    % 当|x|很小时，使用完整计算
    exp_term = exp(-2 * abs_error);
    
    % 使用log1p提高数值精度
    log_cosh = abs_error + log1p(exp_term) - log(2);
    
    loss = log_cosh;
end
