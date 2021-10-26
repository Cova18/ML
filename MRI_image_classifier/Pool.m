function y = Pool(x)
    [xrow, xcol, numFilters] = size(x);
    y = zeros(xrow/2, xcol/2, numFilters);
    
    for k = 1:numFilters
        filter = ones(2)/(2*2);
        image = conv2(x(:,:,k),filter, 'valid');   %calcolo la media su 4 pixel e la assegno a tutti e 4 i pixel
        y(:,:,k) = image(1:2:end, 1:2:end);         % prendo la media e la metto nel pixel che scelgo ogni 4 
    end
end