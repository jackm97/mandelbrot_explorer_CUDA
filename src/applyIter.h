#include "tbb/tbb.h"
#include <Eigen/Dense>
 
using namespace tbb;
using Eigen::Array;
 
class applyIter {
    Array<double, Dynamic, Dynamic> &values, &zr, &zi, &cr, &ci;
    size_t max_iter;
public:
    void operator()( const blocked_range2d<size_t>& r ) const {
        double iters=0,
               R2=1e6,
               zr2=0,
               zi2=0;
        for( size_t i=r.cols().begin(); i!=r.cols().end(); ++i ){ 
           for (size_t j=r.rows().begin(); j!=r.rows().end(); ++j){
                        iters=0;
                        zr2=0;
                        zi2=0;
                        while((zr2+zi2<=R2) && (iters<max_iter)){
                                zi(j,i) = zi(j,i) * zr(j,i);
                                zi(j,i) = zi(j,i) + zi(j,i) + ci(j,i);
                                zr(j,i) = zr2 - zi2 + cr(j,i);
                                zr2 = zr(j,i) * zr(j,i);
                                zi2 = zi(j,i) * zi(j,i);
                                iters++;
                        }
                        values(j,i) = iters;
	   }
	}
    }

    applyIter( Array<double, Dynamic, Dynamic> &values, Array<double, Dynamic, Dynamic> &zr, Array<double, Dynamic, Dynamic> &zi, Array<double, Dynamic, Dynamic> &cr, Array<double, Dynamic, Dynamic> &ci, size_t max_iter) :
        values(values),
    	zr(zr),
	zi(zi),
	cr(cr),
	ci(ci),
	max_iter(max_iter)
    {}
};
