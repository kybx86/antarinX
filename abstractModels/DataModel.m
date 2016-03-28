classdef DataModel
  properties
   
    X_tr
    y_tr
    X_cv
    y_cv
    X_tt
    y_tt
    yraw_tr
    yraw_cv
    yraw_tt
    
  end
  
  methods (Static = false)
    function data_model = ...
      DataModel(X_tr, y_tr, X_cv, y_cv, X_tt, y_tt, yraw_tr, yraw_cv, yraw_tt)
     
      data_model.X_tr    = X_tr;
      data_model.y_tr    = y_tr;
      data_model.X_cv    = X_cv;
      data_model.y_cv    = y_cv;
      data_model.X_tt    = X_tt;
      data_model.y_tt    = y_tt;
      data_model.yraw_tr = yraw_tr;
      data_model.yraw_cv = yraw_cv;
      data_model.yraw_tt = yraw_tt;
      
    end
  end
end