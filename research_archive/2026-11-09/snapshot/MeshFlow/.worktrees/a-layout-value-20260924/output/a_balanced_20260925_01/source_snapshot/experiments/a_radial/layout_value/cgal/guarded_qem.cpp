// A-LAYOUT-VALUE: explicit CGAL 6 classic-plane QEM with prospective guards.
// This file does not change the CGAL algorithm or silently repair output.
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/IO/polygon_mesh_io.h>
#include <CGAL/boost/graph/helpers.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/Surface_mesh_simplification/Edge_collapse_visitor_base.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/GarlandHeckbert_plane_policies.h>
#include <CGAL/intersections.h>
#include <CGAL/version.h>
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using EK = CGAL::Exact_predicates_exact_constructions_kernel;
using Point = K::Point_3;
using Mesh = CGAL::Surface_mesh<Point>;
namespace SMS = CGAL::Surface_mesh_simplification;
namespace PMP = CGAL::Polygon_mesh_processing;
using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point a) {
  return std::chrono::duration<double>(Clock::now()-a).count();
}
static EK::Point_3 exact(const Point& p) { return {p.x(),p.y(),p.z()}; }
static double angle_min(const std::array<Point,3>& p) {
  double ans=180.;
  for(int j=0;j<3;++j) {
    auto a=p[(j+1)%3]-p[j], b=p[(j+2)%3]-p[j];
    double cross=std::sqrt(CGAL::to_double(CGAL::cross_product(a,b).squared_length()));
    ans=std::min(ans, std::atan2(cross,CGAL::to_double(a*b))*180./std::acos(-1.));
  }
  return ans;
}
struct Tri {
  std::array<std::size_t,3> ids;
  std::array<Point,3> p;
  std::array<EK::Point_3,3> e;
  CGAL::Bbox_3 bbox;
  explicit Tri(std::array<std::size_t,3> ids_, std::array<Point,3> p_):ids(ids_),p(p_) {
    for(int i=0;i<3;++i)e[i]=exact(p[i]);
    bbox=p[0].bbox()+p[1].bbox()+p[2].bbox();
  }
  EK::Triangle_3 triangle() const { return {e[0],e[1],e[2]}; }
};

// Intersections are permitted ONLY on the topologically shared simplex.
// Exact constructions avoid classifying an almost-shared point as exact contact.
static bool illegal_intersection(const Tri& a,const Tri& b) {
  if(!CGAL::do_overlap(a.bbox,b.bbox)) return false;
  std::vector<EK::Point_3> shared;
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)
    if(a.ids[i]==b.ids[j])shared.push_back(a.e[i]);
  if(shared.size()>=3)return true;
  auto hit=CGAL::intersection(a.triangle(),b.triangle());
  if(!hit)return false;
  auto allowed=[&](const EK::Point_3& p) {
    if(shared.empty())return false;
    if(shared.size()==1)return p==shared[0];
    return EK::Segment_3(shared[0],shared[1]).has_on(p);
  };
  return std::visit([&](const auto& g)->bool {
    using T=std::decay_t<decltype(g)>;
    if constexpr(std::is_same_v<T,EK::Point_3>)return !allowed(g);
    else if constexpr(std::is_same_v<T,EK::Segment_3>)
      return !allowed(g.source())||!allowed(g.target());
    else if constexpr(std::is_same_v<T,EK::Triangle_3>)
      return !allowed(g.vertex(0))||!allowed(g.vertex(1))||!allowed(g.vertex(2));
    else {
      for(const auto& p:g)if(!allowed(p))return true;
      return false;
    }
  },*hit);
}
struct State {
  Mesh* mesh;
  std::size_t target;
  double min_angle,q05_angle,max_seconds=300.;
  Clock::time_point start;
  std::size_t selected=0,collapsed=0,noncollapsable=0,filters=0;
  std::size_t reject_missing=0,reject_finite=0,reject_degenerate=0,reject_normal=0;
  std::size_t reject_quality=0,reject_intersection=0,filter_pass=0;
  std::size_t collision_pairs=0,exact_pairs=0;
  std::size_t epoch=0,cache_epoch=static_cast<std::size_t>(-1);
  std::vector<Tri> current;
  double filter_seconds=0.;
  std::string stopped="candidate_queue_exhausted";
  void refresh() {
    if(epoch==cache_epoch)return;
    current.clear(); current.reserve(mesh->number_of_faces());
    for(auto f:mesh->faces()) {
      std::array<std::size_t,3> ids;
      std::array<Point,3> points;
      int i=0;
      for(auto v:CGAL::vertices_around_face(mesh->halfedge(f),*mesh)){
        if(i>=3)throw std::runtime_error("non-triangle input");
        ids[i]=v.idx(); points[i]=mesh->point(v);++i;
      }
      if(i!=3)throw std::runtime_error("non-triangle input");
      current.emplace_back(ids,points);
    }
    cache_epoch=epoch;
  }
};
struct Filter {
  std::shared_ptr<State> s;
  template<class Profile>
  std::optional<typename Profile::Point> operator()(const Profile& profile,
                         std::optional<typename Profile::Point> placement) const {
    auto begin=Clock::now();
    struct Timer { State* s; Clock::time_point b; ~Timer(){s->filter_seconds+=seconds(b);} } timer{s.get(),begin};
    ++s->filters;
    if(!placement){++s->reject_missing;return std::nullopt;}
    const Point& q=*placement;
    if(!std::isfinite(q.x())||!std::isfinite(q.y())||!std::isfinite(q.z())){
      ++s->reject_finite;return std::nullopt;
    }
    s->refresh();
    std::size_t v0=profile.v0().idx(),v1=profile.v1().idx();
    std::vector<Tri> changed;
    std::vector<const Tri*> rest;
    changed.reserve(32);rest.reserve(s->current.size());
    for(const Tri& old:s->current) {
      bool h0=false,h1=false;
      for(auto id:old.ids){h0|=(id==v0);h1|=(id==v1);}
      if(h0&&h1)continue; // The two incident faces disappear on collapse.
      if(!h0&&!h1){rest.push_back(&old);continue;}
      auto ids=old.ids; auto pts=old.p;
      for(int k=0;k<3;++k)if(ids[k]==v0||ids[k]==v1){ids[k]=v0;pts[k]=q;}
      Tri next(ids,pts);
      if(CGAL::collinear(next.e[0],next.e[1],next.e[2])){
        ++s->reject_degenerate;return std::nullopt;
      }
      auto old_n=CGAL::cross_product(old.p[1]-old.p[0],old.p[2]-old.p[0]);
      auto new_n=CGAL::cross_product(pts[1]-pts[0],pts[2]-pts[0]);
      if(CGAL::to_double(old_n*new_n)<=0){
        ++s->reject_normal;return std::nullopt;
      }
      if(angle_min(pts)+1e-10<s->min_angle){
        ++s->reject_quality;return std::nullopt;
      }
      changed.push_back(std::move(next));
    }
    for(std::size_t i=0;i<changed.size();++i){
      for(const Tri* other:rest){
        ++s->collision_pairs;
        if(!CGAL::do_overlap(changed[i].bbox,other->bbox))continue;
        ++s->exact_pairs;
        if(illegal_intersection(changed[i],*other)){
          ++s->reject_intersection;return std::nullopt;
        }
      }
      for(std::size_t j=0;j<i;++j) {
        ++s->collision_pairs;
        if(!CGAL::do_overlap(changed[i].bbox,changed[j].bbox))continue;
        ++s->exact_pairs;
        if(illegal_intersection(changed[i],changed[j])){
          ++s->reject_intersection;return std::nullopt;
        }
      }
    }
    ++s->filter_pass;
    return placement;
  }
};
struct Visitor:SMS::Edge_collapse_visitor_base<Mesh> {
  std::shared_ptr<State> s;
  explicit Visitor(std::shared_ptr<State> p):s(std::move(p)){}
  void OnSelected(const Profile&,const std::optional<FT>&,size_type,size_type){++s->selected;}
  void OnCollapsed(const Profile&,const vertex_descriptor&){++s->collapsed;++s->epoch;}
  void OnNonCollapsable(const Profile&){++s->noncollapsable;}
};
struct Stop {
  std::shared_ptr<State> s;
  template<class F,class Profile>
  bool operator()(const F&,const Profile&,std::size_t,std::size_t) const {
    if(s->mesh->number_of_faces()<=s->target){s->stopped="face_target";return true;}
    if(seconds(s->start)>=s->max_seconds){s->stopped="wall_budget";return true;}
    return false;
  }
};
static void selftest() {
  Tri a({0,1,2},{Point(0,0,0),Point(1,0,0),Point(0,1,0)});
  Tri disjoint({3,4,5},{Point(0,0,1),Point(1,0,1),Point(0,1,1)});
  Tri shared_edge_good({1,0,3},{Point(1,0,0),Point(0,0,0),Point(0,-1,0)});
  Tri shared_edge_bad({0,1,3},{Point(0,0,0),Point(1,0,0),Point(.2,.2,0)});
  Tri shared_vertex_good({0,3,4},{Point(0,0,0),Point(-1,0,0),Point(0,-1,0)});
  Tri shared_vertex_bad({0,3,4},{Point(0,0,0),Point(.5,.1,0),Point(.1,.5,0)});
  Tri crossing({3,4,5},{Point(.2,.2,-1),Point(.2,.2,1),Point(.8,.2,0)});
  if(illegal_intersection(a,disjoint)||illegal_intersection(a,shared_edge_good)
     ||!illegal_intersection(a,shared_edge_bad)||illegal_intersection(a,shared_vertex_good)
     ||!illegal_intersection(a,shared_vertex_bad)||!illegal_intersection(a,crossing))
    throw std::runtime_error("exact intersection guard self-test failed");
  std::cout<<"{\"intersection_selftest\":\"PASS\",\"cases\":6}\n";
}
int main(int argc,char** argv){
  try{
    if(argc==2&&std::string(argv[1])=="--self-test"){selftest();return 0;}
    if(argc!=7)throw std::runtime_error("usage: guarded_qem input.off output.off report.json target min_angle_deg q05_deg");
    auto native_start=Clock::now();
    Mesh mesh;
    if(!CGAL::IO::read_polygon_mesh(argv[1],mesh)||!CGAL::is_triangle_mesh(mesh)||mesh.is_empty())
      throw std::runtime_error("cannot read nonempty triangular input");
    if(!CGAL::is_closed(mesh))throw std::runtime_error("input is not closed");
    const std::size_t input_faces=mesh.number_of_faces();
    auto audit_start=Clock::now();
    bool initial_intersection=PMP::does_self_intersect(mesh);
    double preaudit=seconds(audit_start);
    if(initial_intersection)throw std::runtime_error("input has self intersections");
    auto s=std::make_shared<State>();
    s->mesh=&mesh;s->target=std::stoull(argv[4]);s->min_angle=std::stod(argv[5]);s->q05_angle=std::stod(argv[6]);
    s->start=native_start;
    SMS::GarlandHeckbert_plane_policies<Mesh,K> policy(mesh,100.);
    auto collapse_start=Clock::now();
    int removed=SMS::edge_collapse(mesh,Stop{s},
      CGAL::parameters::get_cost(policy.get_cost()).get_placement(policy.get_placement())
        .filter(Filter{s}).visitor(Visitor{s}));
    double collapse_seconds=seconds(collapse_start);
    const std::size_t actual_faces=mesh.number_of_faces();
    mesh.collect_garbage();
    audit_start=Clock::now();
    bool final_intersection=PMP::does_self_intersect(mesh);
    double postaudit=seconds(audit_start);
    bool closed=CGAL::is_closed(mesh);
    if(!CGAL::IO::write_polygon_mesh(argv[2],mesh,CGAL::parameters::stream_precision(17)))
      throw std::runtime_error("cannot write output");
    std::vector<double> angles;
    for(auto f:mesh.faces()){
      std::array<Point,3> p;int i=0;
      for(auto v:CGAL::vertices_around_face(mesh.halfedge(f),mesh))p[i++]=mesh.point(v);
      angles.push_back(angle_min(p));
    }
    std::sort(angles.begin(),angles.end());
    double qloc=.05*(angles.size()-1),qpart=qloc-std::floor(qloc);
    auto qi=static_cast<std::size_t>(std::floor(qloc));
    double q05=angles[qi]*(1.-qpart)+angles[std::min(qi+1,angles.size()-1)]*qpart;
    std::ofstream out(argv[3]);out<<std::setprecision(17)<<std::boolalpha;
    out<<"{\n"
      <<"\"implementation\":\"CGAL_GarlandHeckbert_plane_policies_plus_prospective_guards\",\n"
      <<"\"cgal_version\":\""<<STR(CGAL_VERSION)<<"\",\n"
      <<"\"policy_note\":\"CGAL 6.1.1 exposes classic plane policy; no public plane_and_line policy. Default boundary discontinuity multiplier 100; closed inputs.\",\n"
      <<"\"kernel_cost\":\"Exact_predicates_inexact_constructions_kernel\",\n"
      <<"\"kernel_collision\":\"Exact_predicates_exact_constructions_kernel\",\n"
      <<"\"collision_filter\":\"all prospective changed-vs-rest and changed-vs-changed pairs; bbox broadphase; exact intersection outside shared simplex rejected\",\n"
      <<"\"quality_filter\":\"local changed-face minimum angle only; global q05 is endpoint audit, not per-step guarantee\",\n"
      <<"\"normal_filter\":\"strictly positive old/new face-normal dot product\",\n"
      <<"\"discontinuity_multiplier\":100,\n"
      <<"\"postrepair\":false,\n"
      <<"\"input_faces\":"<<input_faces<<",\"target_faces\":"<<s->target<<",\"actual_faces\":"<<actual_faces<<",\n"
      <<"\"min_angle_threshold_deg\":"<<s->min_angle<<",\"q05_threshold_deg\":"<<s->q05_angle<<",\n"
      <<"\"min_angle_deg\":"<<angles.front()<<",\"angle_q05_deg\":"<<q05<<",\n"
      <<"\"endpoint_quality_pass\":"<<(angles.front()+1e-10>=s->min_angle&&q05+1e-10>=s->q05_angle)<<",\n"
      <<"\"closed\":"<<closed<<",\"final_self_intersection\":"<<final_intersection<<",\n"
      <<"\"stop_reason\":\""<<s->stopped<<"\",\"wall_budget_seconds\":"<<s->max_seconds<<",\n"
      <<"\"selected_candidates\":"<<s->selected<<",\"accepted_collapses\":"<<s->collapsed
      <<",\"topology_noncollapsable\":"<<s->noncollapsable<<",\"removed_edges\":"<<removed<<",\n"
      <<"\"filter_calls\":"<<s->filters<<",\"filter_pass\":"<<s->filter_pass<<",\n"
      <<"\"reject_missing\":"<<s->reject_missing<<",\"reject_nonfinite\":"<<s->reject_finite
      <<",\"reject_degenerate\":"<<s->reject_degenerate<<",\"reject_normal\":"<<s->reject_normal
      <<",\"reject_min_angle\":"<<s->reject_quality<<",\"reject_intersection\":"<<s->reject_intersection<<",\n"
      <<"\"collision_bbox_pairs\":"<<s->collision_pairs<<",\"collision_exact_pairs\":"<<s->exact_pairs<<",\n"
      <<"\"filter_seconds\":"<<s->filter_seconds<<",\"collapse_seconds\":"<<collapse_seconds
      <<",\"input_audit_seconds\":"<<preaudit<<",\"output_audit_seconds\":"<<postaudit
      <<",\"native_total_seconds\":"<<seconds(native_start)<<"\n}\n";
    std::cout<<"actual_faces="<<actual_faces<<" accepted="<<s->collapsed<<" selected="<<s->selected
             <<" seconds="<<seconds(native_start)<<"\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 2;}
}
