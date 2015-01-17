#ifndef DITREE_IO_H_
#define DITREE_IO_H_

#include <fcntl.h>
#include <unistd.h>

#include "common.hpp"
#include "google/protobuf/message.h"
#include "proto/ditree.pb.h"

namespace ditree {

using ::google::protobuf::Message;

inline void MakeTempFilename(string* temp_filename) {
  temp_filename->clear();
  *temp_filename = "/tmp/ditree_test.XXXXXX";
  char* temp_filename_cstr = new char[temp_filename->size()];
  // NOLINT_NEXT_LINE(runtime/printf)
  strcpy(temp_filename_cstr, temp_filename->c_str());
  int fd = mkstemp(temp_filename_cstr);
  CHECK_GE(fd, 0) << "Failed to open a temporary file at: " << *temp_filename;
  close(fd);
  *temp_filename = temp_filename_cstr;
  delete temp_filename_cstr;
}

inline void MakeTempDir(string* temp_dirname) {
  temp_dirname->clear();
  *temp_dirname = "/tmp/ditree_test.XXXXXX";
  char* temp_dirname_cstr = new char[temp_dirname->size()];
  // NOLINT_NEXT_LINE(runtime/printf)
  strcpy(temp_dirname_cstr, temp_dirname->c_str());
  char* mkdtemp_result = mkdtemp(temp_dirname_cstr);
  CHECK(mkdtemp_result != NULL)
      << "Failed to create a temporary directory at: " << *temp_dirname;
  *temp_dirname = temp_dirname_cstr;
  delete temp_dirname_cstr;
}

bool CheckFileExistence(const char* filename);

inline bool CheckFileExistence(const string& filename) {
  return CheckFileExistence(filename.c_str());
}

//bool ReadStringIntMap(const string& filename, map<string, int>& st_int_map) {
//  return ReadStringIntMap(filename.c_str(), st_int_map);
//}
//bool ReadStringIntMap(const char* filename, map<string, int>& st_int_map);

bool ReadProtoFromTextFile(const char* filename, Message* proto);

inline bool ReadProtoFromTextFile(const string& filename, Message* proto) {
  return ReadProtoFromTextFile(filename.c_str(), proto);
}

inline void ReadProtoFromTextFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromTextFile(filename, proto));
}

inline void ReadProtoFromTextFileOrDie(const string& filename, Message* proto) {
  ReadProtoFromTextFileOrDie(filename.c_str(), proto);
}

void WriteProtoToTextFile(const Message& proto, const char* filename);
inline void WriteProtoToTextFile(const Message& proto, const string& filename) {
  WriteProtoToTextFile(proto, filename.c_str());
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto);

inline bool ReadProtoFromBinaryFile(const string& filename, Message* proto) {
  return ReadProtoFromBinaryFile(filename.c_str(), proto);
}

inline void ReadProtoFromBinaryFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromBinaryFile(filename, proto));
}

inline void ReadProtoFromBinaryFileOrDie(const string& filename,
                                         Message* proto) {
  ReadProtoFromBinaryFileOrDie(filename.c_str(), proto);
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename);
inline void WriteProtoToBinaryFile(
    const Message& proto, const string& filename) {
  WriteProtoToBinaryFile(proto, filename.c_str());
}

}  // namespace ditree

#endif   // DITREE_IO_H_
