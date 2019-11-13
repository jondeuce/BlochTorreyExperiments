function call_n_times(f, n_times)

if nargin < 2
    n_times = 1;
end

for i = 1:n_times
    f();
end

end

