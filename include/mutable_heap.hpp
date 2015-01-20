#ifndef DITREE_MUTABLE_HEAP_HPP_
#define DITREE_MUTABLE_HEAP_HPP_

#include "common.hpp"

#include <boost/pending/mutable_queue.hpp>

namespace ditree {

/**
 * Adapted from 
 * github.com/adrienkaiser/MeshSimplification/blob/master/CGAL/property_map.h
 * Also refer to 
 * www.boost.org/doc/libs/1_56_0/boost/property_map/property_map.hpp
 */
template <typename Pair>
struct FirstOfPairPropertyMap
  : public boost::put_get_helper<typename Pair::first_type&,
                                 FirstOfPairPropertyMap<Pair> >
{
  typedef typename Pair::first_type& value_type;
  typedef boost::lvalue_property_map_tag category;

  inline value_type operator[](const Pair& pair) const { 
    return value_type(pair.first); 
  }   
};

// from smallest to largest
struct IdxCntPairComparator {
  bool operator()(
      const pair<uint32, int>& lhs, const pair<uint32, int>& rhs) const {
    return (lhs.second < rhs.second)
        || (lhs.second == rhs.second && lhs.first < rhs.first);
  }
};

//typedef FirstOfPairPropertyMap<IdxCntPair> first_of_pair_prop_map;
typedef boost::mutable_queue<IdxCntPair, vector<IdxCntPair>,
    IdxCntPairComparator, FirstOfPairPropertyMap<IdxCntPair> > 
    IdxCntPairMutableMinHeap;

inline IdxCntPairMutableMinHeap* GetIdxCntPairMutableMinHeap(
    const int max_size) {
  return new IdxCntPairMutableMinHeap(max_size, IdxCntPairComparator(), 
      FirstOfPairPropertyMap<IdxCntPair>());
}

} // namespace ditree

#endif
