export interface Capability {
  status: 'supported' | 'partial' | 'unsupported';
  why: string;
}

export interface NodeAttr {
  shape?: (number | null)[];
  trainable?: boolean;
  weights?: Record<string, { shape: number[]; param_count: number }>;
  [key: string]: any;
}

export interface ModelNode {
  id: string;
  label: string;
  type: string;
  attrs: NodeAttr;
}

export interface ModelEdge {
  source: string;
  target: string;
}

export interface ModelView {
  kind: string;
  name: string;
  nodes: ModelNode[];
  edges: ModelEdge[];
  meta: {
    total_params?: number;
    trainable_params?: number;
    non_trainable_params?: number;
    input_shapes?: (number | null)[][];
    output_shapes?: (number | null)[][];
    [key: string]: any;
  };
  capabilities: Record<string, Capability>;
}