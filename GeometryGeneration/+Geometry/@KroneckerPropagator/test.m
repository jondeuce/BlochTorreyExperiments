function b = test()

    b = true;

    gsiz = [5,6];
    gdim = [5,6];
    b = b && test_kronv_2d(gsiz, gdim);

    gsiz = [5,6,7];
    gdim = [5,6,7];
    b = b && test_kronv_3d(gsiz, gdim);

end

function b = test_kronv_2d(gsiz, gdim)

    b = true; % tests passed flag
    Ax  = randnc(gsiz(1));
    Ay  = randnc(gsiz(2));
    Ayx = Geometry.KroneckerPropagator.kronsum(Ay, Ax);
    
    t = 0.1;
    Ex  = expm(t*Ax);
    Ey  = expm(t*Ay);
    Eyx = expm(t*Ayx);
    
    % Test identity: exp(A ⊕ B) = exp(A) ⊗ exp(B)
    b1 = max(abs(vec(kron(Ey, Ex) - Eyx))) <= 100*eps;
    if ~b1; warning('Kronecker sum to exponential product failed'); end
    b = b && b1;

    % Test action of exp(A ⊕ B):
    G = Geometry.KroneckerPropagator({Ax, Ay}, t, gsiz, gdim);
    X = randnc(gsiz);
    b2 = max(abs(vec(conv(G,X)) - Eyx * vec(X))) <= 100*eps;
    if ~b2; warning('Kronecker sum to exponential product failed'); end
    b = b && b2;

end

function b = test_kronv_3d(gsiz, gdim)

    b = true; % tests passed flag
    Ax   = randnc(gsiz(1));
    Ay   = randnc(gsiz(2));
    Az   = randnc(gsiz(3));
    Azyx = Geometry.KroneckerPropagator.kronsum(Az, Geometry.KroneckerPropagator.kronsum(Ay, Ax));
    
    t = 0.1;
    Ex   = expm(t*Ax);
    Ey   = expm(t*Ay);
    Ez   = expm(t*Az);
    Ezyx = expm(t*Azyx);
    
    % Test identity: exp(A ⊕ B ⊕ C) = exp(A) ⊗ exp(B) ⊗ exp(C)
    b1 = max(abs(vec(kron(Ez, kron(Ey, Ex)) - Ezyx))) <= 100*eps;
    if ~b1; warning('Kronecker sum to exponential product failed'); end
    b = b && b1;

    % Test action of exp(A ⊕ B ⊕ C):
    G = Geometry.KroneckerPropagator({Ax, Ay, Az}, t, gsiz, gdim);
    X = randnc(gsiz);
    b2 = max(abs(vec(conv(G,X)) - Ezyx * vec(X))) <= 100*eps;
    if ~b2; warning('Kronecker sum to exponential product failed'); end
    b = b && b2;
    
end
