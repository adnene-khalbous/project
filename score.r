score = function(y, yc){
  t = table(y, yc)
  diag=0
  for (i in 1:nrow(table(y, yc)))
  {
    diag=diag+t[i,i]
  }
  score=(diag/length(y))*100
}