pause 5
set key autotitle columnhead
set xrange [0:250]
set yrange [40:100]
plot "C:/Users/alpha/Documents/Qt_Projects/clanu/models/best/echo_ADAM.dat" using 1:5 with lines, "C:/Users/alpha/Documents/Qt_Projects/clanu/models/best/echo_ADAM.dat" using 1:6 with lines  
do for [t=0:60]  { 
replot 
pause 2
}
