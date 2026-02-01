classdef MultiHeadAttentionLayer < nnet.layer.Layer
    properties
        NumHeads
        KeySize
        ValueSize
        OutputSize
    end
    
    properties (Learnable)
        WeightsQuery
        WeightsKey
        WeightsValue
        WeightsOutput
    end
    
    methods
        function layer = MultiHeadAttentionLayer(numHeads, keySize, valueSize, inputSize, name)
            % 参数说明:
            %   numHeads: 注意力头数量
            %   keySize: 每个注意力头的键/查询维度
            %   valueSize: 每个注意力头的值维度
            %   inputSize: 输入特征维度 (应与LSTM输出维度匹配)
            
            layer.NumHeads = numHeads;
            layer.KeySize = keySize;
            layer.ValueSize = valueSize;
            layer.OutputSize = inputSize; % 输出维度等于输入维度
            layer.Name = name;
            
            % 正确初始化权重维度
            layer.WeightsQuery = randn([inputSize, keySize, numHeads]) * 0.01;
            layer.WeightsKey = randn([inputSize, keySize, numHeads]) * 0.01;
            layer.WeightsValue = randn([inputSize, valueSize, numHeads]) * 0.01;
            layer.WeightsOutput = randn([numHeads * valueSize, inputSize]) * 0.01;
        end
        
        function Z = predict(layer, X)
            % 输入X维度: [inputSize, sequenceLength, batchSize]
            [inputSize, sequenceLength, batchSize] = size(X);
            Z = zeros(layer.OutputSize, sequenceLength, batchSize);
            
            % 存储各注意力头的输出
            headOutputs = cell(layer.NumHeads, 1);
            
            for head = 1:layer.NumHeads
                % 查询投影: W_q^T * X
                Q = pagemtimes(layer.WeightsQuery(:,:,head), 'transpose', X, 'none');
                
                % 键投影: W_k^T * X
                K = pagemtimes(layer.WeightsKey(:,:,head), 'transpose', X, 'none');
                
                % 值投影: W_v^T * X
                V = pagemtimes(layer.WeightsValue(:,:,head), 'transpose', X, 'none');
                
                % 计算注意力分数: Q * K^T / sqrt(d_k)
                scores = pagemtimes(Q, 'none', K, 'transpose') / sqrt(layer.KeySize);
                
                % 注意力权重 (softmax)
                attentionWeights = softmax(scores, 'DataFormat', 'SCB');
                
                % 加权值: attn * V
                headOutput = pagemtimes(attentionWeights, V);
                headOutputs{head} = headOutput;
            end
            
            % 拼接多头输出
            multiHeadOutput = cat(1, headOutputs{:});
            
            % 输出投影: W_o^T * (多头拼接输出)
            Z = pagemtimes(layer.WeightsOutput, 'transpose', multiHeadOutput, 'none');
        end
    end
end
