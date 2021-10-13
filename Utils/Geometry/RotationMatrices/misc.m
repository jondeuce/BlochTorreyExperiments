V0 = [
-0.002206	-0.235910	0.804715
-0.002983	-0.319785	-0.593662
0.999993	-0.917651	0.000004
];

g0 = V0(:,1);
b0 = V0(:,2);
th0 = angle3D(g0,b0);

g = [0.078453	0.235360	9.649744];
b = [17.000000	-28.727272	-72.551017];

g = g/norm(g);
b = b/norm(b);

B_out = [0.349252	0.936592	-0.028609
-0.937013	0.349261	-0.004859
0.005441	0.028504	0.999579];
B = vecs2basis(g0,b0,g,b);