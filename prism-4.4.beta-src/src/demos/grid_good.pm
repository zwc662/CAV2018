dtmc

formula stay = ((x=6 & y=4) | (x=6 & y=7) | (x=7 & y=7));
formula right = ((x=7 & y=0) | (x=6 & y=1) | (x=7 & y=2) | (x=6 & y=3) | (x=6 & y=5) | (x=6 & y=6) | (x=0 & y=7) | (x=1 & y=7) | (x=2 & y=7) | (x=3 & y=7) | (x=4 & y=7) | (x=5 & y=7));
formula down = ((x=0 & y=0) | (x=1 & y=0) | (x=2 & y=0) | (x=3 & y=0) | (x=5 & y=0) | (x=0 & y=1) | (x=1 & y=1) | (x=2 & y=1) | (x=3 & y=1) | (x=0 & y=2) | (x=1 & y=2) | (x=2 & y=2) | (x=3 & y=2) | (x=0 & y=3) | (x=1 & y=3) | (x=2 & y=3) | (x=3 & y=3) | (x=4 & y=3) | (x=0 & y=4) | (x=1 & y=4) | (x=2 & y=4) | (x=3 & y=4) | (x=4 & y=4) | (x=5 & y=4) | (x=7 & y=4) | (x=0 & y=5) | (x=1 & y=5) | (x=2 & y=5) | (x=3 & y=5) | (x=4 & y=5) | (x=5 & y=5) | (x=7 & y=5) | (x=0 & y=6) | (x=1 & y=6) | (x=2 & y=6) | (x=3 & y=6) | (x=4 & y=6) | (x=5 & y=6) | (x=7 & y=6));
formula left = ((x=4 & y=0) | (x=6 & y=0) | (x=4 & y=1) | (x=5 & y=1) | (x=4 & y=2) | (x=5 & y=2) | (x=6 & y=2) | (x=5 & y=3));
formula up = ((x=7 & y=1) | (x=7 & y=3));

const int x_max = 7;
const int y_max = 7;
const double p = 0.8;
const int x_h_0 = 7;
const int y_h_0 = 7;
const int x_h_1 = 6;
const int y_h_1 = 7;
const int x_l_0 = 6;
const int y_l_0 = 4;
const int x_l_1 = 1;
const int y_l_1 = 6;
const double e = 2.72;
const int x_min = 0;
const int y_min = 0;
const int x_init = 0;
const int y_init = 0;

module grid_good

	x : [0..7];
	y : [0..7];

	[stay] stay=true -> 1 : (x'=x) & (y'=y);
	[right] right=true -> (1-p)/4 : (x'=x) & (y'=y) + p : (x'=(x+1>x_max?x-1:x+1)) & (y'=y) + (1-p)/4 : (x'=x) & (y'=(y+1>y_max?y-1:y+1)) + (1-p)/4 : (x'=(x-1<x_min?x+1:x-1)) & (y'=y) + (1-p)/4 : (x'=x) & (y'=(y-1<y_min?y+1:y-1));
	[down] down=true -> (1-p)/4 : (x'=x) & (y'=y) + (1-p)/4 : (x'=(x+1>x_max?x-1:x+1)) & (y'=y) + p : (x'=x) & (y'=(y+1>y_max?y-1:y+1)) + (1-p)/4 : (x'=(x-1<x_min?x+1:x-1)) & (y'=y) + (1-p)/4 : (x'=x) & (y'=(y-1<y_min?y+1:y-1));
	[left] left=true -> (1-p)/4 : (x'=x) & (y'=y) + (1-p)/4 : (x'=(x+1>x_max?x-1:x+1)) & (y'=y) + (1-p)/4 : (x'=x) & (y'=(y+1>y_max?y-1:y+1)) + p : (x'=(x-1<x_min?x+1:x-1)) & (y'=y) + (1-p)/4 : (x'=x) & (y'=(y-1<y_min?y+1:y-1));
	[up] up=true -> (1-p)/4 : (x'=x) & (y'=y) + (1-p)/4 : (x'=(x+1>x_max?x-1:x+1)) & (y'=y) + (1-p)/4 : (x'=x) & (y'=(y+1>y_max?y-1:y+1)) + (1-p)/4 : (x'=(x-1<x_min?x+1:x-1)) & (y'=y) + p : (x'=x) & (y'=(y-1<y_min?y+1:y-1));

endmodule

init y = 5& x = 0 endinit

