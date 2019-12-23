//
// Created by Myasnikov Vladislav on 10/21/19.
//

#ifndef DXF_SUPPORT_DXFBUILDER_H
#define DXF_SUPPORT_DXFBUILDER_H

#include <string>
#include <list>
#include <map>
#include "core/fileformats/dxf_support/dxfCodes.h"
#include "core/fileformats/dxf_support/dxfDrawing.h"
#include "core/fileformats/dxf_support/objects/dxfObject.h"
#include "core/fileformats/dxf_support/entities/dxfEntity.h"
#include "core/fileformats/dxf_support/blocks/dxfBlock.h"
#include "core/math/vector/vector3d.h"

namespace corecvs {

class DxfBuilder {
public:
    DxfBuilder() = default;

    // Variables
    void setIntVariable(int code, std::string const &name, int value);
    void setDoubleVariable(int code, std::string const &name, double value);
    void setStringVariable(int code, std::string const &name, std::string const &value);
    void set2DVectorVariable(int code, std::string const &name, double x, double y);
    void set3DVectorVariable(int code, std::string const &name, double x, double y, double z);

    // Objects
    void addLayer(DxfLayerObject *object);
    void addLineType(DxfLineTypeObject *object);
    void addBlockRecord(DxfBlockRecordObject *object);

    // Entities
    void addEntity(DxfEntity *entity);

    // Blocks
    void addBlock(DxfBlock *block);

    DxfDrawing getDrawing();

private:
    std::map<std::string,DxfLayerObject*> layers;
    std::map<std::string,DxfBlock*> blocks;
    std::map<std::string,DxfObject*> otherObjects;
    std::map<std::string,std::list<DxfEntity*>> layerEntities;
    std::map<std::string,DxfBlockRecordObject*> blockRecords = {};
};

} // namespace corecvs

#endif //DXF_SUPPORT_DXFBUILDER_H
