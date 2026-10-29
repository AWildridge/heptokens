// Append ordinary uproot-readable TTrees to a complete native generator skim.
// The native CMS products remain the archival authority, including fields not
// duplicated in these convenience views. No detector or generator is run.
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "TFile.h"
#include "TTree.h"
#include "TObjString.h"
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace genbackfill {
struct Links {
  std::vector<unsigned int> offsets, process, product;
  std::vector<long long> index;
  std::vector<unsigned char> state; //0=null,1=unavailable,2=available
  void clear() { offsets.assign(1,0); process.clear(); product.clear(); index.clear(); state.clear(); }
  void branches(TTree &t,const std::string &p) {
    t.Branch((p+"Offset").c_str(),&offsets); t.Branch((p+"Index").c_str(),&index);
    t.Branch((p+"Process").c_str(),&process); t.Branch((p+"Product").c_str(),&product);
    t.Branch((p+"State").c_str(),&state);
  }
  template<class R> void add(const R &r,const std::vector<reco::GenParticle> &target) {
    bool valid=r.isNonnull();
    if(valid && (!r.isAvailable() || r.key()>=target.size() || r.get()!=&target[r.key()]))
      throw std::runtime_error("Reference does not resolve to the retained pruned collection");
    index.push_back(valid?static_cast<long long>(r.key()):-1);
    process.push_back(valid?r.id().processIndex():0); product.push_back(valid?r.id().productIndex():0);
    state.push_back(!valid?0:r.isAvailable()?2:1);
  }
  void end() { offsets.push_back(index.size()); }
};
struct Particles {
  std::vector<int> pdg,status,q3,collision;
  std::vector<unsigned short> flags;
  std::vector<unsigned char> longLived,massConstraint;
  std::vector<double> px,py,pz,energy,pt,eta,phi,mass,x,y,z;
  Links mothers,daughters,lastPruned;
  void clear() {
    pdg.clear();status.clear();q3.clear();collision.clear();flags.clear();longLived.clear();massConstraint.clear();
    px.clear();py.clear();pz.clear();energy.clear();pt.clear();eta.clear();phi.clear();mass.clear();x.clear();y.clear();z.clear();
    mothers.clear();daughters.clear();lastPruned.clear();
  }
  void branches(TTree &t,const std::string &p,bool packed) {
#define GB_BRANCH(key,member) t.Branch((p+key).c_str(),&member)
    GB_BRANCH("pdgId",pdg);GB_BRANCH("status",status);GB_BRANCH("threeCharge",q3);
    GB_BRANCH("statusFlags",flags);GB_BRANCH("longLived",longLived);GB_BRANCH("massConstraint",massConstraint);
    // Store one four-vector representation per collection at getter precision.
    // Packed is natively polar; pruned is natively Cartesian. The native CMS
    // products preserve the original data as well. Derived coordinates belong
    // in the reader rather than duplicating large, high-entropy vectors.
    if(packed) { GB_BRANCH("pt",pt);GB_BRANCH("eta",eta);GB_BRANCH("phi",phi); }
    else { GB_BRANCH("px",px);GB_BRANCH("py",py);GB_BRANCH("pz",pz);GB_BRANCH("energy",energy); }
    GB_BRANCH("mass",mass);
    GB_BRANCH("x_mm",x);GB_BRANCH("y_mm",y);GB_BRANCH("z_mm",z);
    mothers.branches(t,p+"mother");
    if(packed) lastPruned.branches(t,p+"lastPruned");
    else { GB_BRANCH("collisionId",collision);daughters.branches(t,p+"daughter"); }
#undef GB_BRANCH
  }
  template<class P> void particle(const P &p) {
    pdg.push_back(p.pdgId());status.push_back(p.status());q3.push_back(p.threeCharge());
    flags.push_back(p.statusFlags().flags_.to_ulong());longLived.push_back(p.longLived());massConstraint.push_back(p.massConstraint());
    px.push_back(p.px());py.push_back(p.py());pz.push_back(p.pz());energy.push_back(p.energy());
    pt.push_back(p.pt());eta.push_back(p.eta());phi.push_back(p.phi());mass.push_back(p.mass());
    x.push_back(10*p.vx());y.push_back(10*p.vy());z.push_back(10*p.vz());
  }
};
}

long long export_generator_view(const char *filename,const char *summary,const char *metadata) {
  using namespace genbackfill;
  TFile input(filename,"READ");
  if(input.IsZombie()) throw std::runtime_error("Cannot open native generator skim");
  fwlite::Event events(&input);
  const long long expected=events.size();
  TFile output(filename,"UPDATE");
  if(output.IsZombie()||output.Get("gen")) throw std::runtime_error("Fresh gen directory required");
  auto dir=output.mkdir("gen");dir->cd();
  TTree tree("events","All retained packed and pruned particles; native products also preserved");
  tree.SetAutoFlush(-32*1024*1024);
  unsigned int run=0,lumi=0,processId=0;
  unsigned long long event=0,entry=0;
  double weight=0,qScale=0,alphaQCD=0,alphaQED=0,pdfScale=0;
  double pdfX1=0,pdfX2=0,pdf1=0,pdf2=0;
  int pdfId1=0,pdfId2=0,nME=0,nMEFiltered=0;
  bool hasPDF=false;
  std::vector<double> weights,binning;
  std::vector<float> djr;
#define GB_SCALAR(name,type) tree.Branch(#name,&name,#name "/" type)
  GB_SCALAR(run,"i");GB_SCALAR(lumi,"i");GB_SCALAR(event,"l");GB_SCALAR(entry,"l");
  GB_SCALAR(weight,"D");GB_SCALAR(qScale,"D");GB_SCALAR(alphaQCD,"D");GB_SCALAR(alphaQED,"D");
  GB_SCALAR(processId,"i");GB_SCALAR(hasPDF,"O");GB_SCALAR(nME,"I");GB_SCALAR(nMEFiltered,"I");
  GB_SCALAR(pdfScale,"D");GB_SCALAR(pdfX1,"D");GB_SCALAR(pdfX2,"D");GB_SCALAR(pdf1,"D");GB_SCALAR(pdf2,"D");
  GB_SCALAR(pdfId1,"I");GB_SCALAR(pdfId2,"I");
#undef GB_SCALAR
  tree.Branch("weights",&weights);tree.Branch("binningValues",&binning);tree.Branch("DJRValues",&djr);
  Particles packed,pruned;packed.branches(tree,"Packed_",true);pruned.branches(tree,"Pruned_",false);
  fwlite::Handle<std::vector<pat::PackedGenParticle>> hp;
  fwlite::Handle<std::vector<reco::GenParticle>> hg;
  fwlite::Handle<GenEventInfoProduct> hw;
  std::set<std::tuple<unsigned int,unsigned int,unsigned long long>> identities;
  unsigned long long totalPacked=0,totalPruned=0,totalMothers=0,totalDaughters=0;
  for(events.toBegin();!events.atEnd();++events,++entry) {
    auto const &aux=events.eventAuxiliary();run=aux.id().run();lumi=aux.luminosityBlock();event=aux.id().event();
    if(!identities.emplace(run,lumi,event).second) throw std::runtime_error("Duplicate generator event identity");
    hp.getByLabel(events,"packedGenParticles");hg.getByLabel(events,"prunedGenParticles");hw.getByLabel(events,"generator");
    if(!hp.isValid()||!hg.isValid()||!hw.isValid()) throw std::runtime_error("Required generator product absent");
    packed.clear();pruned.clear();
    for(auto const &p:*hp) {
      packed.particle(p);packed.mothers.add(p.motherRef(),*hg);packed.mothers.end();
      packed.lastPruned.add(p.lastPrunedRef(),*hg);packed.lastPruned.end();
    }
    for(auto const &p:*hg) {
      pruned.particle(p);pruned.collision.push_back(p.collisionId());
      for(size_t i=0;i<p.numberOfMothers();++i)pruned.mothers.add(p.motherRef(i),*hg);
      for(size_t i=0;i<p.numberOfDaughters();++i)pruned.daughters.add(p.daughterRef(i),*hg);
      pruned.mothers.end();pruned.daughters.end();
    }
    weight=hw->weight();weights=hw->weights();binning=hw->binningValues();djr=hw->DJRValues();
    qScale=hw->qScale();alphaQCD=hw->alphaQCD();alphaQED=hw->alphaQED();processId=hw->signalProcessID();
    nME=hw->nMEPartons();nMEFiltered=hw->nMEPartonsFiltered();hasPDF=hw->hasPDF();
    pdfId1=pdfId2=0;pdfScale=pdfX1=pdfX2=pdf1=pdf2=0;
    if(hasPDF){auto p=hw->pdf();pdfId1=p->id.first;pdfId2=p->id.second;pdfScale=p->scalePDF;
      pdfX1=p->x.first;pdfX2=p->x.second;pdf1=p->xPDF.first;pdf2=p->xPDF.second;}
    if(tree.Fill()<0)throw std::runtime_error("Cannot fill generator view");
    totalPacked+=packed.pdg.size();totalPruned+=pruned.pdg.size();
    totalMothers+=pruned.mothers.index.size();totalDaughters+=pruned.daughters.index.size();
  }
  if(static_cast<long long>(entry)!=expected)throw std::runtime_error("Native/view event count differs");
  dir->cd();tree.Write();TObjString meta(metadata);meta.Write("schema");
  tree.SetDirectory(nullptr);
  output.Close();input.Close();
  std::ofstream report(summary);
  report<<"{\"events\":"<<entry<<",\"packed\":"<<totalPacked<<",\"pruned\":"<<totalPruned
        <<",\"pruned_mother_links\":"<<totalMothers<<",\"pruned_daughter_links\":"<<totalDaughters
        <<",\"native_products_preserved\":true,\"particle_time_available\":false}"<<std::endl;
  return entry;
}
