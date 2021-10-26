function [W1, W5, Wo] = SGD_MnistConv(W1, W5, Wo, X, D)
    alpha = 0.01;
    beta  = 0.95;
    momentum1 = zeros(size(W1));
    momentum5 = zeros(size(W5));
    momentumo = zeros(size(Wo));
    
    N=length(D);    % Number of images
    
    bsize = 100;    % Definisco i batch (set di immagini)
    blist = 1:bsize:(N-bsize+1);    % Batch indexes
    
    for batch = 1:length(blist)
        dW1 = zeros(size(W1));
        dW5 = zeros(size(W5));
        dWo = zeros(size(Wo));
        
        begin = blist(batch);       
        for k = begin:begin+bsize-1
            k
            x = X(:,:,k);
            y1 = Conv(x, W1);   %Convolution filter % Takes 8 out of the dimensions
            y2 = ReLU(y1);
            y3 = Pool(y2);  % Pooling function (mean pool)
            y4 = reshape(y3, [],1);
            y5 = ReLU(W5*y4);
            v = Wo*y5;
            predicted = Softmax(v); % Result
            
            expected=zeros(2,1);
            expected(sub2ind(size(expected), D(k)+1, 1)) = 1;
            
            % Each step has its own cost function
            cost_function = expected - predicted; % Softmax cost function
            delta = cost_function; 
            dWo = dWo + delta *y5'; % delta is derivate of Softmax*cost_function 
                                    % derivate of the Softmax is 1
         
            cost_function5 = Wo' * delta; % step corresponding to y5
            delta5 = (y5 > 0)  .* cost_function5;
            dW5 = dW5 + delta5*y4';
            
            cost_function4 = W5' * delta5; % step corresponding to y4
            cost_function3 = reshape(cost_function4, size(y3));
            cost_function2 = zeros(size(y2));
            W3 = ones(size(y2))/(2*2);
            
            for c = 1:20
                %kron con W3 e' l'operazione inv a pool (riporta da 10x10x20 a 20x20x20)
                cost_function2(:,:,c) = kron(cost_function3(:,:,c), ones([2 2])) .* W3(:,:,c);      
            end
            
            delta2 = (y2 > 0) .* cost_function2;
            
            delta1_x = zeros(size(W1));         %inizializzo delta_x (9x9x20)
            
            for c = 1:20
                delta1_x(:,:,c) = conv2(x(:,:), rot90(delta2(:,:,c),2),'valid');
            end
            dW1 = dW1 + delta1_x;
           
           
        end
        % Weight correction
        dW1 = dW1 / bsize;
        dW5 = dW5 / bsize;
        dWo = dWo / bsize;
        
        momentum1 = alpha*dW1 + beta*momentum1; %applico un LR corretto del beta momentum 
        W1        = W1 + momentum1;
        
        momentum5 = alpha*dW5 + beta*momentum5;
        W5        = W5 + momentum5;
        
        momentumo = alpha*dWo + beta*momentumo;
        Wo        = Wo + momentumo;
    end
end