%% 自定义Huber损失回归层
classdef huberRegressionLayer < nnet.layer.RegressionLayer
    % huberRegressionLayer   Huber损失回归层
    % 这个层实现了Huber损失函数，对异常值比均方误差更鲁棒
    
    properties
        % Delta参数，控制Huber损失从二次到线性的切换点
        Delta
    end
    
    methods
        function layer = huberRegressionLayer(name, delta)
            % layer = huberRegressionLayer(name, delta) 创建Huber损失回归层
            % 输入:
            %   name - 层名称
            %   delta - Huber损失的delta参数，默认为1.0
            
            if nargin < 2
                delta = 1.0;
            end
            if nargin < 1
                name = 'huber_loss';
            end
            
            layer.Name = name;
            layer.Description = 'Huber损失回归层';
            layer.Type = 'Huber回归层';
            layer.Delta = delta;
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) 返回Huber损失
            % 输入:
            %   layer - 层对象
            %   Y - 网络预测值
            %   T - 目标值
            
            % 确保输入是单精度
            Y = single(Y);
            T = single(T);
            
            % 计算绝对误差
            error = Y - T;
            abs_error = abs(error);
            
            % Huber损失公式
            quadratic = min(abs_error, layer.Delta);
            linear = abs_error - quadratic;
            huber_loss = 0.5 * quadratic.^2 + layer.Delta * linear;
            
            % 计算平均损失
            loss = mean(huber_loss(:));
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % dLdY = backwardLoss(layer, Y, T) 返回Huber损失对Y的梯度
            % 输入:
            %   layer - 层对象
            %   Y - 网络预测值
            %   T - 目标值
            
            % 确保输入是单精度
            Y = single(Y);
            T = single(T);
            
            error = Y - T;
            abs_error = abs(error);
            
            % Huber损失的梯度
            gradient = zeros(size(error), 'single');  % 显式创建单精度数组
            mask = abs_error <= layer.Delta;
            
            % 当|error| <= delta时，梯度为error
            gradient(mask) = error(mask);
            
            % 当|error| > delta时，梯度为delta * sign(error)
            gradient(~mask) = layer.Delta * sign(error(~mask));
            
            % 计算平均梯度（保持单精度）
            dLdY = gradient ./ single(numel(Y));
        end
    end
end
