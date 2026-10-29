// FWLite conversion using the unmodified Delphes writer. No detector response.
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "TFile.h"
#include "classes/DelphesXDRWriter.cc"
#include "classes/DelphesPileUpWriter.cc"
#include <array>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <vector>

long long export_cms_native(const char *input, const char *output,
                            const char *ids_path, const char *progress_path) {
  std::unique_ptr<TFile> file(TFile::Open(input));
  if (!file || file->IsZombie()) throw std::runtime_error("Cannot open GEN input");
  fwlite::Event events(file.get());
  DelphesPileUpWriter writer(output);
  std::ofstream ids(ids_path);
  if (!ids) throw std::runtime_error("Cannot open identity ledger");
  std::set<std::tuple<unsigned int,unsigned int,unsigned long long>> seen;
  long long count=0, particles=0;
  for (events.toBegin(); !events.atEnd(); ++events) {
    const auto &aux=events.eventAuxiliary();
    auto key=std::make_tuple(aux.run(),aux.luminosityBlock(),aux.event());
    if (!seen.insert(key).second) throw std::runtime_error("Duplicate GEN event ID");
    fwlite::Handle<GenEventInfoProduct> weights;
    weights.getByLabel(events,"generator");
    if (!weights.isValid() || weights->weight()!=1.)
      throw std::runtime_error("Missing or non-unit generator weight");
    fwlite::Handle<std::vector<reco::GenParticle>> handle;
    handle.getByLabel(events,"genParticles");
    if (!handle.isValid()) throw std::runtime_error("Missing GEN particles");
    std::vector<const reco::GenParticle*> stable;
    std::map<std::array<double,3>,int> frequencies;
    for (const auto &v:*handle) {
      if (v.status()!=1) continue;
      for (double x:{v.px(),v.py(),v.pz(),v.energy(),v.mass(),v.vx(),v.vy(),v.vz()})
        if (!std::isfinite(x)) throw std::runtime_error("Nonfinite GEN particle");
      stable.push_back(&v);
      ++frequencies[{{10.*v.vx(),10.*v.vy(),10.*v.vz()}}];
    }
    if (stable.empty()) throw std::runtime_error("Empty stable GEN event");
    std::array<double,3> origin;
    int best=0;
    // Canonical Python3 Counter.most_common(1): first occurrence wins ties.
    for (const auto *v:stable) {
      std::array<double,3> xyz={{10.*v->vx(),10.*v->vy(),10.*v->vz()}};
      if (frequencies[xyz]>best) { best=frequencies[xyz]; origin=xyz; }
    }
    for (const auto *v:stable)
      writer.WriteParticle(v->pdgId(),10.*v->vx()-origin[0],10.*v->vy()-origin[1],
                           10.*v->vz()-origin[2],0.,v->px(),v->py(),v->pz(),v->energy());
    writer.WriteEntry();
    ids << "[" << aux.run() << "," << aux.luminosityBlock() << "," << aux.event() << "]\n";
    particles+=stable.size();
    if (++count%1000==0) {
      std::ofstream progress(progress_path);
      progress << "{\"events\":" << count << ",\"particles\":" << particles << "}\n";
    }
  }
  if (count!=events.size() || !count) throw std::runtime_error("Incomplete GEN extraction");
  writer.WriteIndex();
  ids.flush();
  if (!ids) throw std::runtime_error("Identity ledger write failed");
  return particles;
}
