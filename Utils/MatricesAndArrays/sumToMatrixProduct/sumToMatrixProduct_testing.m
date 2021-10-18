tol	=   1e-12;

%% 2D matrix setup
% b(m1,n1) = sum( a(m1,n1,m2,n2) * x(m2,n2) )
% 
% [M1,N1,M2,N2]	=   deal(2,3,4,5);
% a	=   rand(M1,N1,M2,N2);
% x	=   rand(M2,N2);
% b	=   zeros(M1,N1);
% 
% for m1 = 1:M1
%     for n1 = 1:N1
%         for m2 = 1:M2
%             for n2 = 1:N2
%                 b(m1,n1)	=   b(m1,n1) + a(m1,n1,m2,n2) * x(m2,n2);
%             end
%         end
%     end
% end
% 
% A	=   reshape( a, M1*N1, M2*N2 );
% B	=   reshape( A*x(:), M1, N1 );
% 
% e2	=   max(abs(b(:)-B(:)));
% assert( e2 < tol );

%% 3D matrix setup
% b(m1,n1,p1) = sum( a(m1,n1,p1,m2,n2,p2) * x(m2,n2,p2) )

% [M1,N1,P1,M2,N2,P2]	=   deal(5,6,7,8,9,10);
% a	=   rand(M1,N1,P1,M2,N2,P2);
% x	=   rand(M2,N2,P2);
% b	=   zeros(M1,N1,P1);
% for m1 = 1:M1
%     for n1 = 1:N1
%         for p1 = 1:P1
%             for m2 = 1:M2
%                 for n2 = 1:N2
%                     for p2 = 1:P2
%                         b(m1,n1,p1)	=   b(m1,n1,p1) + a(m1,n1,p1,m2,n2,p2) * x(m2,n2,p2);
%                     end
%                 end
%             end
%         end
%     end
% end
% 
% A	=   reshape( a, M1*N1*P1, M2*N2*P2 );
% B	=   reshape( A*x(:), M1, N1, P1 );
% 
% e3	=   max(abs(b(:)-B(:)));
% assert( e3 < tol );

%% 1D simplified case: a(m1,m2) = a(m1-m2)
% b(m1) = sum( a(m1,m2) * x(m2) )

% [M1,M2]	=   deal(5,5);
% a	=   zeros(M1,M2);
% % aa	=   randi(20,M1+M2-1);
% aa	=   (1:(M1+M2-1)).';
% x	=   randi(20,M2,1);
% b	=   zeros(M1,1);
% 
% for m1 = 1:M1
%     for m2 = 1:M2
%         a(m1,m2)	=   aa(m1-m2+M2);
%         b(m1,1)     =   b(m1) + a(m1,m2) * x(m2);
%     end
% end
% 
% A	=   reshape( a, M1, M2 );
% B	=   A*x(:);
% 
% e2	=   max(abs(b(:)-B(:)));
% assert( e2 < tol );

%% 2D simplified case: a(m1,n1,m2,n2) = a(m1-m2,n1-n2)
% b(m1,n1) = sum( a(m1,n1,m2,n2) * x(m2,n2) )

[M1,N1,M2,N2]	=   deal(3,3,3,3);
a	=   zeros(M1,N1,M2,N2);
% aa	=   randi(20,M1+M2-1,N1+N2-1);
aa	=   reshape( (1:(M1+M2-1)*(N1+N2-1)).', M1+M2-1, N1+N2-1 );
x	=   randi(20,M2,N2);
b	=   zeros(M1,N1);

for m1 = 1:M1
    for n1 = 1:N1
        for m2 = 1:M2
            for n2 = 1:N2
                a(m1,n1,m2,n2)	=   aa(m1-m2+M2,n1-n2+N2);
                b(m1,n1)        =   b(m1,n1) + a(m1,n1,m2,n2) * x(m2,n2);
            end
        end
    end
end

A	=   reshape( a, M1*N1, M2*N2 );
B	=   reshape( A*x(:), M1, N1 );

e2	=   max(abs(b(:)-B(:)));
assert( e2 < tol );

%% 3D simplified case: a(m1,n1,p1,m2,n2,p2) = a(m1-m2,n1-n2,p1-p2)
% b(m1,n1,p1) = sum( a(m1,n1,p1,m2,n2,p2) * x(m2,n2,p2) )

% [M1,N1,P1,M2,N2,P2]	=   deal(2,2,2,2,2,2);
% a	=   zeros(M1,N1,P1,M2,N2,P2);
% % aa	=   randi(20,M1+M2-1,N1+N2-1,P1+P2-1);
% aa	=   reshape( (1:(M1+M2-1)*(N1+N2-1)*(P1+P2-1)).', M1+M2-1, N1+N2-1, P1+P2-1 );
% x	=   randi(20,M2,N2,P2);
% b	=   zeros(M1,N1,P1);
% 
% for m1 = 1:M1
%     for n1 = 1:N1
%         for p1 = 1:P1
%             for m2 = 1:M2
%                 for n2 = 1:N2
%                     for p2 = 1:P2
%                         a(m1,n1,p1,m2,n2,p2)	=   aa( m1-m2+M2, n1-n2+N2, p1-p2+P2 );
%                         b(m1,n1,p1)	=   b(m1,n1,p1) + a(m1,n1,p1,m2,n2,p2) * x(m2,n2,p2);
%                     end
%                 end
%             end
%         end
%     end
% end
% 
% A	=   reshape( a, M1*N1*P1, M2*N2*P2 );
% B	=   reshape( A*x(:), M1, N1, P1 );
% 
% e2	=   max(abs(b(:)-B(:)));
% assert( e2 < tol );


























