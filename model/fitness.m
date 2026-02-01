function y = fun(x,Train_xNorm,Train_yNorm,Test_xNorm,Test_y,yopt)
%函数用于计算粒子适应度值


rng default;%固定随机数

NumOfUnits =  fix(x(3))+1; % 隐含层神经元数量 round为四舍五入函数；
%numhidden_units2= fix(x(3))+1;



%  层设置，参数设置
%inputSize = 7;
outputSize = 1;  %数据输出y的维度  




options = trainingOptions('adam', ...
    'MaxEpochs',100, ...
    'GradientThreshold',1,...
    'ExecutionEnvironment','cpu',...
    'InitialLearnRate',x(2), ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',100, ...   %100个epoch后学习率更新
    'LearnRateDropFactor',0.5, ...
    'Shuffle','once',...  % 时间序列长度
    'SequenceLength',1,...
    'MiniBatchSize',128,...
    'L2Regularization', x(1), ... '
    'Verbose',1);
%% lstm

   layers = [    
        sequenceInputLayer([7, 1, 1], 'Name', 'input')

    sequenceFoldingLayer('name','fold')
    convolution2dLayer([2,1],10,'Stride',[1,1],'name','conv1')
    batchNormalizationLayer('name','batchnorm1')
    reluLayer('name','relu1')
    convolution2dLayer([1,1],10,'Stride',[1,1],'name','conv2')
    batchNormalizationLayer('name','batchnorm2')
    reluLayer('name','relu2')
    maxPooling2dLayer([1,3],'Stride',1,'Padding','same','name','maxpool')
    sequenceUnfoldingLayer('name','unfold')
    flattenLayer('name','flatten')
    lstmLayer(NumOfUnits ,'Outputmode','sequence','name','hidden1') 
    dropoutLayer(0.3,'name','dropout_1')

    % 多头注意力层（核心创新）
    MultiHeadAttentionLayer(4, 32, 32, NumOfUnits, 'multihead_attention')
    layerNormalizationLayer('name','attn_norm')

    lstmLayer(NumOfUnits ,'Outputmode','sequence','name','hidden3') 
    dropoutLayer(0.3,'name','dropout_3')
    lstmLayer(NumOfUnits ,'Outputmode','last','name','hidden2') 
    dropoutLayer(0.3,'name','drdiopout_2')
    fullyConnectedLayer(outputSize,'name','fullconnect')   % 全连接层设置（影响输出维度）（cell层出来的输出层） %
    tanhLayer('name','softmax')
    regressionLayer('name','output')];

lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph,'fold/miniBatchSize','unfold/miniBatchSize');
    

%
% 网络训练


net = trainNetwork(Train_xNorm,Train_yNorm,lgraph,options);

Predict_Ynorm = net.predict(Test_xNorm);
Predict_Y  = mapminmax('reverse',Predict_Ynorm',yopt);
Predict_Y = Predict_Y';


rmse_without_update1 = sqrt(mean(abs(Predict_Y-(Test_y)).^2,'ALL'));
y = rmse_without_update1 ;%  cost为目标函数 ，目标函数为rmse
end
