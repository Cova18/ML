function y = Conv(x, W)
    % Applies the chosen filter W to the x image passed as input
    [wrow, wcol, numFilters] = size(W);
    [xrow, xcol, ~         ] = size(x);

    yrow = xrow - wrow + 1;
    ycol = xcol - wcol + 1;

    y = zeros(yrow, ycol, numFilters);

    for k = 1:numFilters
        filter = W(:,:,k);
        filter = rot90(squeeze(filter),2);  % Needed for compatibility between the filter and the image
        y(:,:,k) = conv2(x, filter, 'valid');   
        % Bidimentional convolution between x and filter
        % The 'valid' parameter takes only the results computed without the
        % zero-padded edges
    end
end