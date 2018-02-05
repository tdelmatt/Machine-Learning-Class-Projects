function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);
m = size(X,1);
%maxcentroid = initialize a column vector with xrows storing max centroid in each case

%comparecentroids = ...initialize a column vector with centroids columns
comparecentroids = zeros(K, 1);

%for all rows in X (coordinates of training examples)
for i = 1 : m,
	%tempX = rowvector(i) of X
	tempX = X(i, :);
	%for all rows in centroids (coordinates)
	for j = 1 : K,
		%tempcentroid = rowvector(j) of centroid
		tempcentroid = centroids(j, :);
		%comparecentroids(j) = ||tempcentroid - tempx||^2   ----norm squared------
		comparecentroids(j) = (tempcentroid - tempX) * (tempcentroid - tempX)';
	end;
	comparecentroids;
	%end
	[val(i) idx(i)] = min(comparecentroids);
end;
		


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%







% =============================================================

end

